"""
recorder.py — Asynchronous HDF5 snapshot writer for Project Prismatic Monism.

Writes simulation snapshots to HDF5 at an adaptive recording rate
(10 Hz for steps 0–5,000; 1 Hz thereafter) using a dedicated
ThreadPoolExecutor worker thread to eliminate I/O jitter from the
1000 Hz physics loop.

Schema (development_specifications.md §4.2):
  /sim_N              scalar uint16   — node count
  /sim_seed           scalar uint32   — random seed
  /sim_dt             scalar float32  — integration timestep
  /sim_lambda_c       scalar float32  — vacuum constant
  /sim_max_steps      scalar uint32   — configured max step count
  /snapshots/step     1D uint32  (M)       — timestep per snapshot
  /snapshots/S         2D float32 (M×N)    — state vector per snapshot
  /snapshots/W_i       1D uint16  (K)      — target (row) indices
  /snapshots/W_j       1D uint16  (K)      — source (col) indices
  /snapshots/W_val     1D float32 (K)      — weight values
  /snapshots/W_ptr     1D uint32  (M+1)    — pointer into W_* arrays
  /snapshots/active_edges  1D uint32 (M)
  /snapshots/delta_W   1D float32 (M)
  /snapshots/delta_S   1D float32 (M)
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np


# ── Adaptive Recording Rate ───────────────────────────────────────────────

def should_record(step):
    """
    Determine whether a snapshot should be recorded at the given step.

    Adaptive rate (development_specifications.md §4.1):
      steps 0 – 5,000  → 10 Hz  (every 100th step)
      steps > 5,000    → 1 Hz   (every 1000th step)

    Args:
        step: int — current physics tick index (0-based)

    Returns:
        bool — True if this step should be recorded
    """
    if step <= 5000:
        return step % 100 == 0
    else:
        return step % 1000 == 0


# ── Internal Helpers ──────────────────────────────────────────────────────

_SENTINEL = object()  # signals the writer worker to stop


def _extract_sparse_edges(W_np):
    """
    Extract non-zero edges from a numpy weight matrix.

    Returns three parallel arrays suitable for the HDF5 sparse storage:
    W_i (target/row), W_j (source/col), W_val (weight).

    Args:
        W_np: (N, N) numpy float32 — weight matrix

    Returns:
        (rows, cols, values): tuple of numpy arrays (uint16, uint16, float32)
    """
    rows, cols = np.nonzero(W_np)
    values = W_np[rows, cols]
    return (
        rows.astype(np.uint16),
        cols.astype(np.uint16),
        values.astype(np.float32),
    )


def _put_sentinel(snapshot_queue):
    """Place the sentinel value on the queue to signal the worker to exit."""
    snapshot_queue.put(_SENTINEL)


# ── Public API ────────────────────────────────────────────────────────────

def create_file(filepath, N, seed, dt, lambda_c, max_steps):
    """
    Create and initialize an HDF5 file with simulation metadata.

    All snapshot datasets are created with chunked storage and unlimited
    maxshape so they can be extended as the run progresses.

    Args:
        filepath: str or Path — output HDF5 file path
        N: int — node count
        seed: int — PRNG seed
        dt: float — integration timestep
        lambda_c: float — vacuum constant
        max_steps: int — configured safety cap

    Returns:
        h5py.File — open file handle (caller is responsible for closing)
    """
    f = h5py.File(filepath, "w")

    # ── Simulation metadata (scalar attributes) ──
    f.create_dataset("sim_N", data=np.uint16(N))
    f.create_dataset("sim_seed", data=np.uint32(seed))
    f.create_dataset("sim_dt", data=np.float32(dt))
    f.create_dataset("sim_lambda_c", data=np.float32(lambda_c))
    f.create_dataset("sim_max_steps", data=np.uint32(max_steps))

    # ── Snapshot datasets (resizable) ──
    f.create_dataset(
        "snapshots/step",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint32,
        chunks=(1024,),
    )
    f.create_dataset(
        "snapshots/S",
        shape=(0, N),
        maxshape=(None, N),
        dtype=np.float32,
        chunks=(64, N),
    )
    f.create_dataset(
        "snapshots/W_i",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(4096,),
    )
    f.create_dataset(
        "snapshots/W_j",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(4096,),
    )
    f.create_dataset(
        "snapshots/W_val",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=(4096,),
    )
    f.create_dataset(
        "snapshots/W_ptr",
        shape=(1,),
        maxshape=(None,),
        dtype=np.uint32,
        chunks=(1024,),
    )
    f["snapshots/W_ptr"][0] = 0  # first pointer starts at 0

    f.create_dataset(
        "snapshots/active_edges",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint32,
        chunks=(1024,),
    )
    f.create_dataset(
        "snapshots/delta_W",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=(1024,),
    )
    f.create_dataset(
        "snapshots/delta_S",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
        chunks=(1024,),
    )

    return f


def enqueue_snapshot(snapshot_queue, step, S, W, delta_W, delta_S):
    """
    Copy arrays and enqueue a snapshot for asynchronous HDF5 write.

    Called from the main physics thread. Copies the JAX arrays to numpy
    and extracts sparse edge data before placing the payload on the
    thread-safe queue.

    Args:
        snapshot_queue: queue.Queue — shared queue to the writer worker
        step: int — current physics tick index
        S: jax Array (N,) float32 — state vector
        W: jax Array (N,N) float32 — weight matrix
        delta_W: float — mean absolute weight change this tick
        delta_S: float — mean absolute state change this tick
    """
    # Copy JAX arrays to numpy (forces transfer to CPU if on GPU)
    S_np = np.array(S, dtype=np.float32)
    W_np = np.array(W, dtype=np.float32)

    active_edges = int(np.count_nonzero(W_np))
    rows, cols, values = _extract_sparse_edges(W_np)

    payload = {
        "step": step,
        "S": S_np,
        "W_rows": rows,
        "W_cols": cols,
        "W_vals": values,
        "active_edges": active_edges,
        "delta_W": float(delta_W),
        "delta_S": float(delta_S),
    }
    snapshot_queue.put(payload)


def writer_worker(snapshot_queue, h5file):
    """
    Worker function that runs on the dedicated ThreadPoolExecutor thread.

    Drains the queue, writing each snapshot to HDF5. Exits when it
    receives the sentinel value.

    This function must ONLY be called from a single worker thread.
    h5py file handles are not safe for concurrent access.

    Args:
        snapshot_queue: queue.Queue — shared queue fed by enqueue_snapshot
        h5file: h5py.File — open HDF5 file handle
    """
    while True:
        payload = snapshot_queue.get()

        if payload is _SENTINEL:
            # Drain any remaining items before exiting
            snapshot_queue.task_done()
            break

        try:
            _write_one_snapshot(h5file, payload)
        finally:
            snapshot_queue.task_done()


def _write_one_snapshot(h5file, payload):
    """
    Write a single snapshot payload into the HDF5 file.

    Resizes each extendable dataset by one row (or by the number of
    new sparse entries for the W_* datasets), then writes the new data
    into the last position.

    Args:
        h5file: h5py.File
        payload: dict as produced by enqueue_snapshot
    """
    # ── Step index ──
    ds_step = h5file["snapshots/step"]
    m = ds_step.shape[0]  # current number of snapshots
    ds_step.resize((m + 1,))
    ds_step[m] = payload["step"]

    # ── State vector S ──
    ds_S = h5file["snapshots/S"]
    ds_S.resize((m + 1, ds_S.shape[1]))
    ds_S[m, :] = payload["S"]

    # ── Sparse edges ──
    k_prev = int(h5file["snapshots/W_ptr"][m])  # where this snapshot starts
    nz = len(payload["W_vals"])
    k_new = k_prev + nz

    for ds_name in ("W_i", "W_j", "W_val"):
        ds = h5file[f"snapshots/{ds_name}"]
        ds.resize((k_new,))

    h5file["snapshots/W_i"][k_prev:k_new] = payload["W_rows"]
    h5file["snapshots/W_j"][k_prev:k_new] = payload["W_cols"]
    h5file["snapshots/W_val"][k_prev:k_new] = payload["W_vals"]

    # Extend W_ptr by one entry (M+1 total after this write)
    ds_ptr = h5file["snapshots/W_ptr"]
    # ds_ptr already has shape (m+1) with the last entry being k_prev
    ds_ptr.resize((m + 2,))
    ds_ptr[m + 1] = k_new

    # ── Active edges ──
    ds_ae = h5file["snapshots/active_edges"]
    ds_ae.resize((m + 1,))
    ds_ae[m] = payload["active_edges"]

    # ── Delta metrics ──
    ds_dW = h5file["snapshots/delta_W"]
    ds_dW.resize((m + 1,))
    ds_dW[m] = payload["delta_W"]

    ds_dS = h5file["snapshots/delta_S"]
    ds_dS.resize((m + 1,))
    ds_dS[m] = payload["delta_S"]


def start_writer(snapshot_queue, h5file):
    """
    Start the dedicated writer thread via ThreadPoolExecutor.

    Returns a future that can be used to wait for completion.

    Args:
        snapshot_queue: queue.Queue
        h5file: h5py.File

    Returns:
        concurrent.futures.Future
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(writer_worker, snapshot_queue, h5file)
    # Shutdown the executor so it releases resources after the worker finishes
    executor.shutdown(wait=False)
    return future


def close_file(snapshot_queue, h5file, writer_future):
    """
    Gracefully shut down the HDF5 writer.

    1. Places the sentinel on the queue.
    2. Waits for the worker thread to drain and exit.
    3. Closes the HDF5 file.

    Args:
        snapshot_queue: queue.Queue
        h5file: h5py.File
        writer_future: concurrent.futures.Future — from start_writer
    """
    _put_sentinel(snapshot_queue)
    writer_future.result()  # block until worker exits
    h5file.close()