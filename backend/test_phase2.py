"""
Phase 2 Integration Test — HDF5 Recording & Round-Trip Verification
=====================================================================
Runs N=4 for 10,000 ticks with adaptive-rate HDF5 recording, then
verifies the file has the correct number of snapshots and W_ptr
properly indexes sparse edges.

Also exercises the read_hdf5 module to confirm round-trip correctness.

Run from the project root with:
  env\Scripts\python.exe backend\test_phase2.py
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, "backend")

import jax.numpy as jnp
import numpy as np
from engine import initialize, tick, DT, EPSILON_LOG
from recorder import (
    create_file,
    enqueue_snapshot,
    start_writer,
    close_file,
    should_record,
)
import queue


def test_hdf5_recording():
    """N=4, 10,000 ticks: verify HDF5 recording with adaptive rate."""
    print("=" * 60)
    print("  Phase 2 HDF5 Recording Test: N=4, 10,000 ticks")
    print("=" * 60)

    N = 4
    seed = 42
    max_steps = 10_000

    S, W, lambda_c = initialize(N, seed)

    # Create a temp file so we don't clutter the project directory
    tmpdir = tempfile.mkdtemp(prefix="monist_test_")
    filepath = os.path.join(tmpdir, f"simulation_N{N}_test.h5")
    print(f"\nHDF5 output: {filepath}")
    print(f"Config: N={N}, seed={seed}, dt={DT}, lambda_c={lambda_c:.6f}, max_steps={max_steps}")

    # ── Initialize HDF5 ──
    h5file = create_file(filepath, N, seed, DT, lambda_c, max_steps)
    snap_queue = queue.Queue()
    writer_future = start_writer(snap_queue, h5file)

    # ── Run simulation ──
    snapshot_count = 0
    expected_snapshots = sum(1 for s in range(max_steps + 1) if should_record(s))
    print(f"Expected snapshots: {expected_snapshots}")

    t_start = time.perf_counter()

    for step in range(max_steps + 1):  # 0 to 10000 inclusive
        if step > 0:
            S, W, active_edges, delta_W, delta_S = tick(S, W, DT, lambda_c, EPSILON_LOG)
        else:
            active_edges = int(jnp.count_nonzero(W))
            delta_W = 0.0
            delta_S = 0.0

        if should_record(step):
            snapshot_count += 1
            enqueue_snapshot(snap_queue, step, S, W, delta_W, delta_S)

    # ── Close (send sentinel, wait for worker, close file) ──
    close_file(snap_queue, h5file, writer_future)

    t_elapsed = time.perf_counter() - t_start
    print(f"Simulation finished: {t_elapsed:.2f}s ({max_steps / t_elapsed:.0f} ticks/sec)")
    print(f"Snapshots enqueued: {snapshot_count}")

    # ── Verify HDF5 contents ──
    print("\n--- Verifying HDF5 contents ---")
    import h5py

    with h5py.File(filepath, "r") as f:
        # Metadata
        assert f["sim_N"][()] == N, f"sim_N mismatch: {f['sim_N'][()]} != {N}"
        assert f["sim_seed"][()] == seed
        assert abs(f["sim_dt"][()] - DT) < 1e-10
        assert abs(f["sim_lambda_c"][()] - lambda_c) < 1e-10
        assert f["sim_max_steps"][()] == max_steps
        print("  [OK] Metadata correct")

        # Snapshot count
        M = f["snapshots/step"].shape[0]
        assert M == expected_snapshots, f"Snapshot count mismatch: {M} != {expected_snapshots}"
        print(f"  [OK] Snapshot count: {M} (expected {expected_snapshots})")

        # Steps are monotonically increasing
        steps = f["snapshots/step"][:]
        assert np.all(np.diff(steps) > 0), "Steps are not monotonically increasing"
        assert steps[0] == 0, f"First step should be 0, got {steps[0]}"
        assert steps[-1] <= max_steps
        print(f"  [OK] Steps monotonic: {steps[0]} -> {steps[-1]}")

        # S shape and values
        S_all = f["snapshots/S"][:]
        assert S_all.shape == (M, N), f"S shape mismatch: {S_all.shape} != ({M}, {N})"
        for i in range(M):
            s_sum = np.sum(S_all[i])
            assert abs(s_sum - 1.0) < 1e-5, f"sum(S) != 1 at snapshot {i}: {s_sum}"
            assert np.all(S_all[i] >= -1e-7), f"Negative S at snapshot {i}"
        print(f"  [OK] S invariants: sum=1, non-negative for all {M} snapshots")

        # W_ptr
        W_ptr = f["snapshots/W_ptr"][:]
        assert W_ptr.shape[0] == M + 1, f"W_ptr shape: {W_ptr.shape[0]} != {M + 1}"
        assert W_ptr[0] == 0, f"W_ptr[0] should be 0, got {W_ptr[0]}"
        assert np.all(np.diff(W_ptr) >= 0), "W_ptr not monotonically non-decreasing"
        total_edges = W_ptr[-1]
        print(f"  [OK] W_ptr: {M + 1} entries, total non-zero edges stored: {total_edges}")

        # Verify W_ptr indexing: for each snapshot t, W_ptr[t] to W_ptr[t+1]
        # are the sparse edges for that snapshot
        W_i = f["snapshots/W_i"][:]
        W_j = f["snapshots/W_j"][:]
        W_val = f["snapshots/W_val"][:]
        assert len(W_i) == total_edges
        assert len(W_j) == total_edges
        assert len(W_val) == total_edges

        active_edges_all = f["snapshots/active_edges"][:]
        assert active_edges_all.shape[0] == M

        # Spot-check: for each snapshot, active_edges matches the W_ptr span
        for t in range(M):
            span = W_ptr[t + 1] - W_ptr[t]
            assert span == active_edges_all[t], (
                f"Snapshot {t}: W_ptr span {span} != active_edges {active_edges_all[t]}"
            )
        print(f"  [OK] W_ptr spans match active_edges for all snapshots")

        # Check no self-edges in stored data
        for t in range(M):
            start = W_ptr[t]
            end = W_ptr[t + 1]
            rows = W_i[start:end]
            cols = W_j[start:end]
            if len(rows) > 0:
                self_edges = np.sum(rows == cols)
                assert self_edges == 0, f"Snapshot {t}: {self_edges} self-edges found"
        print(f"  [OK] No self-edges in any stored snapshot")

        # delta_W and delta_S
        dW_all = f["snapshots/delta_W"][:]
        dS_all = f["snapshots/delta_S"][:]
        assert dW_all.shape[0] == M
        assert dS_all.shape[0] == M
        # First snapshot (step 0) should have delta=0
        assert dW_all[0] == 0.0 and dS_all[0] == 0.0, "First snapshot deltas should be 0"
        print(f"  [OK] delta_W/delta_S datasets correct shape, first entry 0")

    print(f"\n[PASS] HDF5 recording verified successfully.")
    return filepath


def test_round_trip(filepath):
    """Verify round-trip correctness using read_hdf5 module."""
    print("\n" + "=" * 60)
    print("  Round-Trip Verification (via read_hdf5)")
    print("=" * 60)

    from read_hdf5 import load_snapshot, print_summary, print_snapshot

    print_summary(filepath)

    # Load a few snapshots at different stages
    # Step 0 (initial), step 5000 (end of 10 Hz zone), step 10000 (final)
    # Since steps are recorded at multiples of 100/1000, find the exact indices
    import h5py
    with h5py.File(filepath, "r") as f:
        all_steps = f["snapshots/step"][:]
        print(f"\nAvailable step samples: {all_steps[:5].tolist()}...{all_steps[-3:].tolist()}")
        # Find snapshot nearest to step 5000
        idx_5000 = np.searchsorted(all_steps, 5000)
        if idx_5000 < len(all_steps):
            step_5k = int(all_steps[idx_5000])
        else:
            step_5k = int(all_steps[-1])

    print(f"\nLoading snapshot at step 0:")
    S0, W0 = load_snapshot(filepath, 0)
    print_snapshot(S0, W0)
    assert abs(np.sum(S0) - 1.0) < 1e-5, "sum(S) at step 0 != 1"
    assert np.sum(np.abs(np.diag(W0))) < 1e-7, "Non-zero diagonal at step 0"
    print("  [OK] Snapshot 0 invariants hold")

    print(f"\nLoading snapshot at step {step_5k}:")
    S_mid, W_mid = load_snapshot(filepath, step_5k)
    print_snapshot(S_mid, W_mid)
    assert abs(np.sum(S_mid) - 1.0) < 1e-5, f"sum(S) at step {step_5k} != 1"
    print(f"  [OK] Snapshot {step_5k} invariants hold")

    final_step = int(all_steps[-1])
    print(f"\nLoading snapshot at step {final_step}:")
    S_end, W_end = load_snapshot(filepath, final_step)
    print_snapshot(S_end, W_end)
    assert abs(np.sum(S_end) - 1.0) < 1e-5, f"sum(S) at step {final_step} != 1"
    print(f"  [OK] Snapshot {final_step} invariants hold")

    print(f"\n[PASS] Round-trip verification: all snapshots load correctly.")


if __name__ == "__main__":
    print("PHASE 2 INTEGRATION TEST — Project Prismatic Monism")
    print("=" * 60)

    filepath = test_hdf5_recording()
    test_round_trip(filepath)

    # Cleanup temp file
    os.unlink(filepath)
    os.rmdir(os.path.dirname(filepath))
    print(f"\n[OK] Temporary test file cleaned up.")

    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS COMPLETE")
    print("=" * 60)