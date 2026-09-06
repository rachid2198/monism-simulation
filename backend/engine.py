"""
engine.py — Core JAX physics engine for Project Prismatic Monism.

Every function implements the exact mathematics from theory_specifications.md.
This is a pure dynamical system, not a neural network training loop.
"""

import jax
import jax.numpy as jnp

# ── Numerical Constants (theory_specifications.md §5) ────────────────────

DT = 0.001               # Fundamental integration timestep (Planck tick)
EPSILON_LOG = 1e-12      # float32 underflow guard for log(P + ε) only
FIM_FLOAT_GUARD = 1e-16  # Hardware NaN guard on FIM denominator only


# ── Sub-functions (JIT-inlined by parent; no standalone JIT) ─────────────

def compute_P(S, W):
    """
    Generative prediction vector.
    Equation: z_i = Σ_j W_ij · S_j; P_i = softmax(z)_i
    
    Args:
        S: (N,) float32 — state probability vector, ΣS_i = 1
        W: (N,N) float32 — hollow weight matrix, W_ii = 0
    Returns:
        P: (N,) float32 — predicted state distribution, ΣP_i = 1, P_i > 0
    """
    z = W @ S
    return jax.nn.softmax(z)


def compute_F_smooth(S, W, eps_log):
    """
    Smooth free energy (no L1 terms — those are handled by the proximal operator).
    Equation: F_smooth = -Σ_i S_i · ln(P_i + ε_log)
    
    Args:
        S: (N,) float32
        W: (N,N) float32
        eps_log: float — underflow guard for log
    Returns:
        scalar float32
    """
    P = compute_P(S, W)
    return -jnp.sum(S * jnp.log(P + eps_log))


def compute_F_total(S, W, lambda_c, eps_log):
    """
    Total free energy (L1 complexity cost included).
    Equation: F_total = Σ_i S_i · (-ln(P_i + ε_log) + λ_c · Σ_j |W_ij|)
    
    Args:
        S: (N,) float32
        W: (N,N) float32
        lambda_c: float — vacuum constant ln(N)/N
        eps_log: float — underflow guard for log
    Returns:
        F_total: scalar float32
        F_per_node: (N,) float32 — per-node free energy F_i
    """
    P = compute_P(S, W)
    log_term = -jnp.log(P + eps_log)
    l1_term = lambda_c * jnp.sum(jnp.abs(W), axis=1)  # sum over outgoing edges
    F_per_node = S * (log_term + l1_term)
    return jnp.sum(F_per_node), F_per_node


def compute_FIM(S, W, P, lambda_c):
    """
    Total Fisher Information Metric (per-edge diagonal approximation).
    Equation: FIM_ij = S_j² · P_i · (1 - P_i) + λ_c²
    
    The S_j² factor means edges from high-mass source nodes have higher
    information content. The λ_c² term is the Fisher information of the
    Laplace(0, 1/λ_c) vacuum prior.
    
    Args:
        S: (N,) float32
        W: (N,N) float32 — unused (kept for consistent interface)
        P: (N,) float32
        lambda_c: float
    Returns:
        fim: (N,N) float32 — per-edge FIM, guarded against hardware underflow
    """
    # Broadcasting: S[None, :]² is (1, N), P[:, None] * (1-P)[:, None] is (N, 1)
    # Result is (N, N) where fim[i,j] depends on S_j (source) and P_i (target)
    fim = (S[None, :] ** 2) * P[:, None] * (1.0 - P[:, None]) + lambda_c ** 2
    fim = jnp.maximum(fim, FIM_FLOAT_GUARD)
    return fim


def replicator_step(S, f):
    """
    Natural gradient projection onto the probability simplex.
    Equation: dS_i/dt = S_i · (f_i - Σ_k S_k · f_k)
    
    Automatically preserves ΣS_i = 1 (mass conservation).
    
    Args:
        S: (N,) float32
        f: (N,) float32 — fitness vector f_i = -∂F_total/∂S_i
    Returns:
        dS_dt: (N,) float32
    """
    f_mean = jnp.sum(S * f)
    return S * (f - f_mean)


def proximal_weight_update(W, driving_force, FIM, S, dt, lambda_c):
    """
    Natural gradient step + proximal soft-thresholding (the Pruner).
    
    Step 1: W_temp = W + dt · D / FIM  (natural gradient on smooth loss)
    Step 2: θ_ij = dt · S_i · λ_c / FIM_ij  (per-edge L1 threshold)
    Step 3: W_new = sign(W_temp) · max(0, |W_temp| - θ)  (soft-threshold)
    Step 4: W_ii = 0  (hollow matrix constraint)
    
    The threshold scales with S_i (target node mass), so edges into inert
    nodes are cheap to sever.
    
    Args:
        W: (N,N) float32 — current weights
        driving_force: (N,N) float32 — D_ij = -∂F_smooth/∂W_ij
        FIM: (N,N) float32 — per-edge Fisher Information
        S: (N,) float32 — current state
        dt: float — integration timestep
        lambda_c: float — vacuum constant
    Returns:
        W_new: (N,N) float32 — updated weights, hollow, with exact zeros
    """
    # Natural gradient step
    W_temp = W + dt * driving_force / FIM
    
    # Proximal soft-threshold: L1 penalty handled analytically
    # S[:, None] broadcasts S_i across columns (all source nodes for target i)
    threshold = dt * S[:, None] * lambda_c / FIM
    W_new = jnp.sign(W_temp) * jnp.maximum(0.0, jnp.abs(W_temp) - threshold)
    
    # Enforce hollow matrix: zero the diagonal
    W_new = W_new * (1.0 - jnp.eye(W.shape[0]))
    return W_new


# ── Main Tick Function (JIT-compiled) ────────────────────────────────────

@jax.jit
def tick(S, W, dt, lambda_c, eps_log):
    """
    One complete physics tick: state evolution + weight evolution.
    
    Pipeline (theory_specifications.md §4):
      1. Compute P = softmax(W @ S)                              [Eq. §3.1]
      2. Compute F_total, autodiff to get fitness f = -∂F/∂S    [Eq. §4.1]
      3. Replicator step: dS/dt = S · (f - S·f)                  [Eq. §4.1]
      4. Euler integrate, clamp negatives, normalize             [Eq. §4.1]
      5. Compute F_smooth, autodiff to get driving force D       [Eq. §4.2]
      6. Compute FIM per edge                                    [Eq. §3.3]
      7. Natural gradient + proximal update for W                [Eq. §4.2]
      8. Return new state, diagnostics
    
    Args:
        S: (N,) float32 — state vector at time t
        W: (N,N) float32 — weight matrix at time t, W_ii = 0
        dt: float — integration timestep (0.001)
        lambda_c: float — vacuum constant ln(N)/N
        eps_log: float — underflow guard for log
    Returns:
        S_new: (N,) float32 — state at time t+dt, ΣS_i = 1
        W_new: (N,N) float32 — weights at time t+dt, W_ii = 0
        active_edges: int32 — count of non-zero W entries
        delta_W: float32 — mean absolute weight change
        delta_S: float32 — mean absolute state change
    """
    P = compute_P(S, W)
    
    # ── State Update (Fast Dynamics) ──
    # Autodiff through the full F_total scalar w.r.t. S
    f = -jax.grad(lambda s: compute_F_total(s, W, lambda_c, eps_log)[0])(S)
    dS_dt = replicator_step(S, f)
    S_new = S + dt * dS_dt
    S_new = jnp.maximum(S_new, 0.0)    # clamp negatives
    S_new = S_new / jnp.sum(S_new)     # normalize to Σ=1
    
    # ── Weight Update (Slow Dynamics) ──
    # Autodiff through F_smooth w.r.t. W
    driving_force = -jax.grad(compute_F_smooth, argnums=1)(S, W, eps_log)
    FIM = compute_FIM(S, W, P, lambda_c)
    W_new = proximal_weight_update(W, driving_force, FIM, S, dt, lambda_c)
    
    # ── Diagnostics ──
    active_edges = jnp.count_nonzero(W_new)
    delta_W = jnp.mean(jnp.abs(W_new - W))
    delta_S = jnp.mean(jnp.abs(S_new - S))
    
    return S_new, W_new, active_edges, delta_W, delta_S


# ── Initialization ──────────────────────────────────────────────────────

def initialize(N, seed=42):
    """
    Initialize the universe at t=0.
    
    States: uniform S_i = 1/N
    Weights: IID Normal(0, 1), W_ii = 0
    Lambda_c: ln(N)/N
    
    Args:
        N: int — number of nodes
        seed: int — PRNG seed
    Returns:
        S: (N,) float32 — uniform state distribution
        W: (N,N) float32 — random hollow weight matrix
        lambda_c: float — vacuum constant
    """
    key = jax.random.PRNGKey(seed)
    S = jnp.ones(N, dtype=jnp.float32) / N
    W = jax.random.normal(key, (N, N), dtype=jnp.float32)
    W = W * (1.0 - jnp.eye(N, dtype=jnp.float32))  # hollow matrix
    lambda_c = jnp.log(jnp.float32(N)) / N
    return S, W, lambda_c