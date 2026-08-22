# System Specification: Project Prismatic Monism

## 1. System Overview & Core Invariants

Project Prismatic Monism is a continuous-time, self-regulating informational physics engine. The universe is modeled as a single directed graph whose topology and internal state evolution are driven by the minimization of Variational Free Energy (F) via Natural Gradient Descent.

### Core Invariants
- Monism: Matter (states) and spacetime (weights) are two attributes of a single unified graph.
- State Mass Conservation: The sum of state probabilities across all nodes strictly equals 1.0 at all times (Sum(S) == 1.0).
- Zero Static Hyper-parameters: No global learning rates, static time-delays, or manual tuning dials exist. Temporal separation and structural plasticity emerge dynamically from Information Geometry (Fisher Information).

---

## 2. Variable Definitions & Data Structures

### Network Dimensions
- N: Total number of nodes in the graph (Integer, e.g., N = 4 for initial test, N = 1000 for full run).

### Core Data Vectors & Matrices
- State Vector S:
  - Type: Float32 array of shape (N,)
  - Constraint: Sum(S) == 1.0, and 0.0 <= S_i <= 1.0 for all i.
  - Description: Represents the distribution of energy/attention across the graph.

- Topological Weight Matrix W:
  - Type: Float32 matrix of shape (N, N)
  - Constraint: Continuous continuous-valued matrix W_ij where row i represents target node, column j represents source node.
  - Special Value: W_ij == 0.0 represents a severed connection (topological boundary).

- Generative Prediction Vector P:
  - Type: Float32 array of shape (N,)
  - Constraint: Sum(P) == 1.0, and P_i > 0.0 for all i.
  - Description: The network's internal prediction of state probabilities.

---

## 3. Initial Conditions (t = 0)

1. States S:
   S_i = 1.0 / N  (for all i in 0..N-1)

2. Weights W:
   W_ij = Sample from Normal(mean=0.0, stddev=0.1)  (for all i, j in 0..N-1)

3. Constants for Numerical Damping:
   - EPSILON_LOG = 1e-12 (Prevents log(0) errors)
   - EPSILON_FIM = 1e-6  (Damping constant for Fisher Information division)
   - PRUNE_THRESHOLD = 0.001 (Threshold for sparsity rendering)

---

## 4. Mathematical Engine & Formulas (Python/ASCII Syntax)

### 4.1. Generative Prediction Vector (P)
Given states S and weights W:
1. Compute raw activations z:
   z_i = sum(W_ij * S_j for j in 0..N-1)
2. Apply Softmax to compute probability distribution P:
   P_i = exp(z_i) / sum(exp(z_k) for k in 0..N-1)

### 4.2. Local Shannon Entropy (H)
Measures the current predictive uncertainty of the network:
H = - sum(P_k * log(P_k + EPSILON_LOG) for k in 0..N-1)

### 4.3. Local Variational Free Energy (F_i)
Calculated per node i to balance accuracy against complexity:
F_i = - S_i * log(P_i + EPSILON_LOG) + H * sum(abs(W_ij) for j in 0..N-1)

### 4.4. Fast Dynamics: State Derivative (dS/dt)
States move along a probability simplex via Replicator Dynamics (Natural Gradient on a simplex):

1. Compute raw fitness (negative partial derivative of F with respect to S_i):
   f_i = log(P_i + EPSILON_LOG) + S_i * (1.0 / (P_i + EPSILON_LOG)) - H_penalty_term
2. Compute system average fitness:
   f_mean = sum(S_k * f_k for k in 0..N-1)
3. Calculate dS_i/dt:
   dS_i/dt = S_i * (f_i - f_mean)

### 4.5. Slow Dynamics: Weight Derivative (dW/dt)
Weights move via Natural Gradient Descent using a diagonal Fisher Information Matrix (FIM) approximation:

1. Calculate diagonal Fisher Information for edge W_ij:
   FIM_W_ij = (S_j ** 2) * P_i * (1.0 - P_i) + EPSILON_FIM

2. Calculate raw Euclidean partial derivative (dF_i / dW_ij):
   grad_W_ij = - S_i * (1.0 - P_i) * S_j - P_i * (log(P_i + EPSILON_LOG) + H) * S_j * sum(abs(W_im) for m in 0..N-1) + H * sign(W_ij)

3. Calculate final natural gradient weight derivative dW_ij/dt:
   dW_ij/dt = - (1.0 / FIM_W_ij) * grad_W_ij

---

## 5. Integration Step & Guardrails (per time-step dt)

For time-step dt (e.g., dt = 0.001):

1. Update States:
   S_next = S + dt * dS_dt

2. Enforce State Guardrails (Non-negativity & Normalization):
   S_clamped = max(0.0, S_next)
   S_normalized = S_clamped / sum(S_clamped)

3. Update Weights:
   W_next = W + dt * dW_dt

4. Subgradient Rule for sign(W_ij):
   If W_ij == 0.0, then sign(W_ij) = 0.0.

5. Active Edge Masking (for sparse output stream):
   is_active_edge(i, j) = abs(W_ij) >= PRUNE_THRESHOLD

---

## 6. Execution Architecture

- Backend Engine: Python using JAX compiled with JIT (`@jax.jit`).
- Execution Frequency: Physics loop executes at 1000 Hz.
- Streaming Bridge: FastAPI WebSocket server broadcasting binary array buffers at 60 Hz.
- Binary Packet Format: 
  - Float32Array containing [States (length N), Active Edges Count, Active Edge Triples (Source, Target, Weight)]
- Frontend Engine: Three.js rendering active edges as dynamic lines (opacity = abs(W)) and nodes as spheres (scale/glow = S_i).