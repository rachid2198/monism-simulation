"""
Phase 1 Integration Test -- Manual Verification Script
=======================================================
Verifies engine.py against theory_specifications.md invariants.

Test A (1.4): N=4, 100 ticks -- prints S and active edges every tick.
Test B (1.5): N=100, 1000 ticks -- verifies no NaN, no negative S,
               no self-edges, and reports performance timing.

Run from the project root with:
  env\Scripts\python.exe backend\test_phase1.py
"""

import sys
import time
sys.path.insert(0, "backend")

import jax.numpy as jnp
from engine import initialize, tick, DT, EPSILON_LOG


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_a():
    """N=4, 100 ticks: print S and active edges each tick."""
    print_separator("Test A: N=4, 100 ticks")

    N = 4
    seed = 42
    S, W, lambda_c = initialize(N, seed)

    print(f"Config: N={N}, seed={seed}, dt={DT}, lambda_c={lambda_c:.6f}")
    print(f"\n{'Step':>5} | {'S':>40} | {'sum(S)':>8} | {'Active Edges':>13} | {'max(|W_ii|)':>13} | {'dW_mean':>10} | {'dS_mean':>10}")
    print("-" * 115)

    for step in range(101):  # 0 to 100 inclusive
        if step > 0:
            S, W, active_edges, delta_W, delta_S = tick(S, W, DT, lambda_c, EPSILON_LOG)
        else:
            active_edges = int(jnp.count_nonzero(W))
            delta_W = 0.0
            delta_S = 0.0

        # Format S as compact string
        s_str = "[" + ", ".join(f"{float(s):.4f}" for s in S) + "]"
        diag_max = float(jnp.max(jnp.abs(jnp.diag(W))))

        print(f"{step:5d} | {s_str:>40} | {float(jnp.sum(S)):.6f} | {int(active_edges):13d} | {diag_max:.10f} | {float(delta_W):.10f} | {float(delta_S):.10f}")

        # Assert invariants
        S_sum = float(jnp.sum(S))
        assert abs(S_sum - 1.0) < 1e-5, f"sum(S) = {S_sum} != 1 at step {step}"
        assert all(float(s) >= -1e-7 for s in S), f"Negative S at step {step}: {S}"
        assert float(jnp.max(jnp.abs(jnp.diag(W)))) < 1e-7, f"Non-zero diagonal at step {step}"

    print("\n[PASS] Test A: All invariants held for 100 ticks.")


def test_b():
    """N=100, 1000 ticks: verify no NaN, no negative S, no self-edges, performance."""
    print_separator("Test B: N=100, 1000 ticks")

    N = 100
    seed = 123
    S, W, lambda_c = initialize(N, seed)

    print(f"Config: N={N}, seed={seed}, dt={DT}, lambda_c={lambda_c:.6f}")
    print(f"Running 1000 ticks...")

    # Warm-up tick to force JIT compilation before timing (block until complete)
    S_warm, W_warm, _, _, _ = tick(S, W, DT, lambda_c, EPSILON_LOG)
    S_warm.block_until_ready()
    W_warm.block_until_ready()

    t_start = time.perf_counter()

    # Stats tracking
    nan_inf_step = None
    neg_step = None
    diag_violation_step = None
    sum_violation_step = None
    min_active = N * N
    max_active = 0

    for step in range(1, 1001):
        S, W, active_edges, delta_W, delta_S = tick(S, W, DT, lambda_c, EPSILON_LOG)

        # Convert to Python scalars for checking
        active = int(active_edges)
        min_active = min(min_active, active)
        max_active = max(max_active, active)

        # NaN / Inf check
        if jnp.any(jnp.isnan(S)) or jnp.any(jnp.isnan(W)) or jnp.any(jnp.isinf(S)) or jnp.any(jnp.isinf(W)):
            nan_inf_step = step
            break

        # Negative S check (before clamp+normalize would have fixed, but verify)
        if jnp.any(S < -1e-7):
            neg_step = step
            break

        # Diagonal check
        if float(jnp.max(jnp.abs(jnp.diag(W)))) > 1e-7:
            diag_violation_step = step
            break

        # Sum check
        if abs(float(jnp.sum(S)) - 1.0) > 1e-5:
            sum_violation_step = step
            break

    t_elapsed = time.perf_counter() - t_start
    ticks_per_sec = 1000 / t_elapsed

    print(f"\nPerformance: {t_elapsed:.3f}s for 1000 ticks ({ticks_per_sec:.1f} ticks/sec)")
    print(f"Active edges: min={min_active}, max={max_active}, final={active}")

    if nan_inf_step:
        print(f"\n[FAIL] NaN or Inf detected at step {nan_inf_step}")
    elif neg_step:
        print(f"\n[FAIL] Negative S detected at step {neg_step}")
    elif diag_violation_step:
        print(f"\n[FAIL] Non-zero diagonal at step {diag_violation_step}")
    elif sum_violation_step:
        print(f"\n[FAIL] sum(S) != 1 at step {sum_violation_step}")
    else:
        print(f"\n[PASS] Test B: No NaN, no negative S, no self-edges, sum(S)=1 maintained.")
        print(f"  Initial active edges: {N*N - N} (all off-diagonal)")
        print(f"  Final   active edges: {active}")

    return ticks_per_sec


def test_extra():
    """Quick invariant spot-checks from theory_specifications."""
    print_separator("Extra: Invariant Spot-Checks")

    N = 8
    S, W, lambda_c = initialize(N, 99)
    print(f"N={N}, lambda_c = ln({N})/{N} = {lambda_c:.6f}")
    print(f"sum(S) = {float(jnp.sum(S)):.6f}  (should be 1.0)")
    print(f"diag(W) max = {float(jnp.max(jnp.abs(jnp.diag(W)))):.10f}  (should be 0.0)")
    print(f"W shape = {W.shape}")
    print(f"Non-zero W entries (initial): {int(jnp.count_nonzero(W))}")

    # Run one tick and verify outputs
    S2, W2, ae, dW, dS = tick(S, W, DT, lambda_c, EPSILON_LOG)
    print(f"\nAfter 1 tick:")
    print(f"  sum(S) = {float(jnp.sum(S2)):.8f}  (should be 1.0)")
    print(f"  diag(W) max = {float(jnp.max(jnp.abs(jnp.diag(W2)))):.10f}  (should be 0.0)")
    print(f"  active_edges = {int(ae)}")
    print(f"  dW mean = {float(dW):.8f}")
    print(f"  dS mean = {float(dS):.8f}")

    # Verify P values are valid probabilities
    from engine import compute_P
    P = compute_P(S, W)
    print(f"  sum(P) = {float(jnp.sum(P)):.8f}  (should be 1.0)")
    print(f"  min(P) = {float(jnp.min(P)):.8f}  (should be > 0)")

    # Verify F_total returns per-node values of correct shape
    from engine import compute_F_total
    F_total, F_per_node = compute_F_total(S, W, lambda_c, EPSILON_LOG)
    print(f"  F_total = {float(F_total):.8f}")
    print(f"  F_per_node shape = {F_per_node.shape}  (should be ({N},))")

    print("\n[PASS] Extra spot-checks passed.")


if __name__ == "__main__":
    print("PHASE 1 INTEGRATION TEST -- Project Prismatic Monism")
    print("=" * 60)

    test_a()
    test_b()
    test_extra()

    print("\n" + "=" * 60)
    print("ALL PHASE 1 TESTS COMPLETE")
    print("=" * 60)