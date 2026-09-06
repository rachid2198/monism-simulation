"""
read_hdf5.py — Mini HDF5 reader for Project Prismatic Monism snapshots.

Loads individual snapshots from a simulation HDF5 file and prints
summary statistics. Standalone script usable for offline analysis.

Usage:
  env\\Scripts\\python.exe backend\\read_hdf5.py <filepath> [step_index]
"""

import sys

import h5py
import numpy as np


def load_snapshot(filepath, step):
    """
    Load the state vector S and dense weight matrix W for a specific
    simulation step from the HDF5 file.

    The HDF5 stores W in sparse format (W_i, W_j, W_val, W_ptr).
    This function reconstructs the dense (N, N) matrix for the
    requested step.

    Args:
        filepath: str or Path — path to the simulation HDF5 file
        step: int — the exact timestep to load (must be present in
                   /snapshots/step)

    Returns:
        S: (N,) float32 numpy array — state vector
        W: (N, N) float32 numpy array — dense weight matrix

    Raises:
        ValueError: if the requested step is not found in the file
    """
    with h5py.File(filepath, "r") as f:
        # Find the snapshot index for this step
        all_steps = f["snapshots/step"][:]
        matches = np.where(all_steps == step)[0]
        if len(matches) == 0:
            available = all_steps.tolist()
            raise ValueError(
                f"Step {step} not found. Available steps: "
                f"{available[:5]} ... {available[-5:]} "
                f"({len(available)} total)"
            )
        idx = matches[0]

        N = int(f["sim_N"][()])

        # Read state vector
        S = f["snapshots/S"][idx, :].copy()

        # Read sparse edges for this snapshot
        W_ptr = f["snapshots/W_ptr"][:]
        start = W_ptr[idx]
        end = W_ptr[idx + 1]

        if end > start:
            rows = f["snapshots/W_i"][start:end]
            cols = f["snapshots/W_j"][start:end]
            vals = f["snapshots/W_val"][start:end]
        else:
            rows = np.array([], dtype=np.uint16)
            cols = np.array([], dtype=np.uint16)
            vals = np.array([], dtype=np.float32)

        # Reconstruct dense matrix
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(len(vals)):
            W[int(rows[i]), int(cols[i])] = vals[i]

    return S, W


def print_summary(filepath):
    """
    Print a summary of the HDF5 file: metadata, snapshot count, and
    overall statistics.

    Args:
        filepath: str or Path
    """
    with h5py.File(filepath, "r") as f:
        N = int(f["sim_N"][()])
        seed = int(f["sim_seed"][()])
        dt = float(f["sim_dt"][()])
        lambda_c = float(f["sim_lambda_c"][()])
        max_steps = int(f["sim_max_steps"][()])

        M = f["snapshots/step"].shape[0]
        steps = f["snapshots/step"][:]
        total_edges_stored = f["snapshots/W_ptr"][-1]

        # Active edges summary
        active = f["snapshots/active_edges"][:]
        S_final = f["snapshots/S"][-1, :]

    print(f"=== HDF5 Summary: {filepath} ===")
    print(f"  Nodes (N):      {N}")
    print(f"  Seed:           {seed}")
    print(f"  dt:             {dt}")
    print(f"  lambda_c:       {lambda_c:.6f}")
    print(f"  Max steps:      {max_steps}")
    print(f"  Snapshots (M):  {M}")
    print(f"  Step range:     {steps[0]} -> {steps[-1]}")
    print(f"  Sparse edges stored: {total_edges_stored}")
    if M > 0:
        print(f"  Active edges:   min={active.min()}, max={active.max()}, "
              f"final={active[-1]}")
        print(f"  Final S:        min={S_final.min():.6f}, max={S_final.max():.6f}, "
              f"sum={S_final.sum():.8f}")


def print_snapshot(S, W):
    """
    Print detailed stats for a single snapshot.

    Args:
        S: (N,) float32 numpy array
        W: (N, N) float32 numpy array
    """
    N = len(S)
    nonzero = int(np.count_nonzero(W))
    total_possible = N * N - N  # exclude diagonal
    sparsity = 1.0 - (nonzero / total_possible) if total_possible > 0 else 1.0

    print(f"  S:              {np.array2string(S, precision=6, suppress_small=True)}")
    print(f"  sum(S):         {S.sum():.8f}")
    print(f"  Non-zero W:     {nonzero} / {total_possible} ({sparsity:.2%} sparse)")
    print(f"  max(|W|):       {np.max(np.abs(W)):.6f}")
    print(f"  max(|W_diag|):  {np.max(np.abs(np.diag(W))):.10f}")
    if nonzero > 0 and nonzero <= 20:
        # Print the actual edges for small graphs
        rows, cols = np.nonzero(W)
        print(f"  Edges:")
        for r, c in zip(rows, cols):
            print(f"    [{r}->{c}] = {W[r, c]:+.6f}")


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_hdf5.py <filepath> [step]")
        print("  filepath: path to simulation HDF5 file")
        print("  step:     (optional) specific timestep to load")
        sys.exit(1)

    filepath = sys.argv[1]

    if len(sys.argv) >= 3:
        step = int(sys.argv[2])
        print(f"Loading snapshot at step {step}...")
        try:
            S, W = load_snapshot(filepath, step)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print_snapshot(S, W)
    else:
        print_summary(filepath)