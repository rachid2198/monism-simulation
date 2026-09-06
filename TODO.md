# TODO: Project Prismatic Monism

Implementation order is designed to validate each layer before building the next. Every phase ends with a testable artifact.

**Status legend:** `[Pending]` = not started | `[Review]` = agent completed, awaiting human verification | `[Done]` = human confirmed

---

## Phase 1: Core Physics Engine (JAX)

- [Done] **1.1** Activate the existing `env/` virtual environment and install JAX (CPU first, GPU later), numpy, h5py. The environment already exists in `F:\Research\Philosophy\monist-simulation\env\`.
- [Done] **1.2** Implement `engine.py`:
  - `compute_P(S, W)` — softmax prediction
  - `compute_F_smooth(S, W, eps_log)` — smooth free energy (no L1)
  - `compute_F_total(S, W, lambda_c, eps_log)` — returns scalar and per-node `F_i`
  - `compute_FIM(S, W, P, lambda_c)` — per-edge total Fisher Information with float guard
  - `replicator_step(S, f)` — natural gradient projection on simplex
  - `proximal_weight_update(W, driving_force, FIM, S, dt, lambda_c)` — natural gradient + soft-threshold + hollow matrix
  - `tick(S, W, dt, lambda_c, eps_log)` — full JIT-compiled step
- [Done] **1.3** Implement `initialize(N, seed)` — uniform S, Normal(0,1) W, hollow matrix, λ_c.
- [Done] **1.4** Write a manual integration test: run N=4 for 100 ticks, print S and active edge count each tick. Verify S sums to 1, W diagonal is 0, sparsity emerges or stabilizes.
- [Done] **1.5** Test with N=100, 1000 ticks. Verify no NaN, no negative S, no self-edges, performance timing.

---

## Phase 2: HDF5 Recording

- [Pending] **2.1** Implement `recorder.py`:
  - `create_file(filepath, N, seed, dt, lambda_c, max_steps)` — initializes HDF5 with metadata
  - `enqueue_snapshot(snapshot_queue, step, S, W)` — copies arrays and enqueues for async write
  - `writer_worker(snapshot_queue, file)` — runs on a dedicated `ThreadPoolExecutor` worker thread, draining the queue and writing HDF5 snapshots
  - `close_file(file, worker_future)` — drains remaining queue, joins worker, closes file
  - **The ThreadPoolExecutor is mandatory, not optional.**
- [Pending] **2.2** Implement adaptive-rate logic: 10 Hz for steps 0–5000, 1 Hz after.
- [Pending] **2.3** Write an integration test: run N=4 for 10,000 ticks, verify the HDF5 file has the correct number of snapshots and W_ptr properly indexes sparse edges.
- [Pending] **2.4** Write a mini `read_hdf5.py` script that loads a snapshot and prints summary stats. Confirm round-trip correctness.

---

## Phase 3: Simulation Lifecycle & Run Loop

- [Pending] **3.1** Implement `lifecycle.py`:
  - `SimulationState` dataclass: status, step, S, W, lambda_c, config, delta buffer
  - `run_loop` — while status == RUNNING, call tick, check snapshot interval, call recorder, check equilibrium, check max_steps
  - Equilibrium detection: 500-tick sliding window of ΔW and ΔS, threshold 1e-6
- [Pending] **3.2** Write an integration test: start, run 2000 ticks, pause, check S sum = 1, resume, run to equilibrium or max_steps.
- [Pending] **3.3** Test equilibrium auto-pause: run a non-evolving config (e.g., N=2 with trivial S) and confirm it auto-pauses.

---

## Phase 4: FastAPI Server (Backend API)

- [Pending] **4.1** Implement `main.py`:
  - `POST /simulation/start` — validate params, call initialize, start run loop in background thread/asyncio task
  - `POST /simulation/pause` — set status to PAUSED
  - `POST /simulation/resume` — set status to RUNNING, restart loop
  - `POST /simulation/reset` — set status to IDLE, clear state
  - `GET /simulation/status` — return JSON: status, step, N, seed, active_edges, delta_W, delta_S
- [Pending] **4.2** Implement `websocket_handler.py`:
  - Accept WebSocket connections at `/ws`
  - Read S and active edges from the simulation state at the throttled rate
  - Encode binary packets: `[uint32 N][float32*N S][uint32 edge_count][uint16*2+float32]*edge_count`
  - Dynamic throttle: >200K edges → 5 Hz, 50K–200K → 15 Hz, ≤50K → 60 Hz
  - On simulation pause/stop, send a final packet with `edge_count = 0xFFFFFFFF` as a sentinel
- [Pending] **4.3** Test with a WebSocket client (e.g., `websocat` or a simple Python script):
  - Start simulation, connect WS, verify packet decode for first 100 packets
  - Verify throttle changes as edge count drops

---

## Phase 5: Pause & Analyze Endpoint

- [Pending] **5.1** Implement `analysis.py`:
  - `load_snapshot(filepath, step)` — read S and W for a specific snapshot from HDF5
  - `louvain_clustering(W_abs)` — run Louvain community detection on absolute weight matrix (undirected)
  - `spectral_coordinates(W_abs, dims=3)` — compute first 3 non-trivial Laplacian eigenvectors
- [Pending] **5.2** Implement `POST /simulation/analyze`:
  - Read HDF5 snapshot for requested step
  - Run requested analysis (louvain, spectral, or both)
  - Return JSON: `{clusters: [...], spectral_coords: [[x,y,z], ...]}`
- [Pending] **5.3** Test with a paused simulation: call `/analyze`, verify cluster assignments are sensible (at least all nodes assigned, no empty clusters).

---

## Phase 6: Frontend Shell (Vite + React)

- [Pending] **6.1** Scaffold Vite + React + TypeScript project:
  - `pnpm create vite frontend --template react-ts` (or npm)
  - Install Three.js, d3-force-3d
- [Pending] **6.2** Implement `<ControlPanel>`:
  - N slider (4–1000, default 100)
  - Seed text input + "Random" checkbox
  - Max Steps input (default 100,000)
  - Start, Pause, Resume, Reset buttons (enabled/disabled per sim status)
  - Status indicator text
  - "Pause & Analyze" button
- [Pending] **6.3** Implement `useSimulation` hook:
  - `start(params)`, `pause()`, `resume()`, `reset()`, `getStatus()`, `analyze(step, type)`
  - Calls backend REST endpoints via `fetch`
- [Pending] **6.4** Implement `useWebSocket` hook:
  - Connect to `ws://localhost:8000/ws` when simulation is running
  - Parse binary packets into `Float32Array` for S and typed arrays for edges
  - Store in `useRef` (NOT React state)
  - Track connection status, auto-reconnect on disconnect
- [Pending] **6.5** Test: Start simulation via UI, verify status updates, verify WebSocket connects and receives data.

---

## Phase 7: 2D Adjacency Matrix Canvas

- [Pending] **7.1** Implement `<AdjacencyMatrix>`:
  - `<canvas>` element with `useRef`
  - On each data arrival: sort node indices by S descending
  - Fill `ImageData` where pixel (i, j) = color from `|W[sorted_i][sorted_j]|`
  - Use `putImageData` for fast rendering
  - Handle N scaling: for N ≤ canvas_size, each cell = 1+ pixels; for N > canvas_size, downsample
- [Pending] **7.2** Implement "Pause & Analyze" re-sort:
  - When analysis results arrive, re-sort the matrix by cluster assignment instead of S
  - Draw cluster boundary lines
- [Pending] **7.3** Test with live data: verify the matrix updates, hot corner appears at top-left, re-sort on analysis.

---

## Phase 8: Three.js 3D Force Graph

- [Pending] **8.1** Implement Web Worker `forceLayout.worker.ts`:
  - Import d3-force-3d
  - Initialize with N nodes in random 3D positions
  - On message (new edges): update force links with `|W_ij|` as strength
  - Run simulation ticks in a loop, post node positions back as `Float32Array[3*N]` at ~60 Hz
  - Handle alpha cooling and restart on new data
- [Pending] **8.2** Implement `<ThreeScene>`:
  - Three.js `WebGLRenderer` attached to a `<div>` container
  - `InstancedMesh` for nodes: sphere geometry, per-instance color and scale from S
  - `LineSegments` `BufferGeometry` for edges: rebuild geometry when edge count changes; update vertex positions each frame from Web Worker positions
  - Camera: orbit controls, auto-rotate (toggle)
  - During plasma phase: render only a random subset of edges (cap at ~50K) to maintain 60 FPS
- [Pending] **8.3** Wire Web Worker ↔ Three.js:
  - `useRef` holds the shared `Float32Array` of node positions
  - Worker posts positions, main thread updates InstancedMesh matrices and edge BufferGeometry each render frame
  - WebSocket data → Worker (new edges/weights); Worker → Main (positions)
- [Pending] **8.4** Test: verify 3D graph renders, nodes size/color reflects S, edges appear, force layout converges.

---

## Phase 9: Integration & Polish

- [Pending] **9.1** End-to-end test: Start simulation from UI, watch both views update, pause, analyze, verify cluster re-sort, resume.
- [Pending] **9.2** Performance profiling:
  - Verify physics loop maintains 1000 Hz at N=1000 (GPU) or acceptable CPU rate
  - Verify WebSocket throttling activates at the right edge counts
  - Verify Three.js maintains 60 FPS at each tier
- [Pending] **9.3** Error handling:
  - Backend: validation errors on invalid params, graceful shutdown
  - Frontend: WebSocket disconnect/reconnect, API error toasts
- [Pending] **9.4** GPU setup: Switch from `jax[cpu]` to `jax[cuda12]`, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` before importing JAX to reserve VRAM for the browser's WebGL contexts, test on RTX 3060.
- [Pending] **9.5** Seed reproducibility test: Run N=4, seed=42 twice, confirm identical HDF5 outputs.

---

## Phase 10: Documentation & Notebooks

- [Pending] **10.1** Write `README.md`: how to install, run, use the UI, interpret results.
- [Pending] **10.2** Create `notebooks/analysis_demo.ipynb`: loading HDF5, plotting S over time, plotting active edge count over time, running Louvain, visualizing clusters.
- [Pending] **10.3** Add inline code comments and docstrings to all backend functions.

---

## Optional / Deferred Features

These are described in `development_specifications.md` Section 13. They may be implemented after the core MVP is complete and stable:

1. **Markov Blanket Detection** — conditional-independence boundary identification in paused snapshots.
2. **Self-Model Discovery** — detecting subgraphs whose internal P mirrors global S.
3. **Physics-Isomorphism Checks** — verifying whether network dynamics replicate known physical laws.
4. **Timeline Scrubber** — UI control for scrubbing through saved HDF5 history.
5. **Spectral Clustering** — alternative to Louvain for the analysis view.
6. **Jupyter Notebooks** (advanced) — additional deep-dive analysis notebooks.
7. **Ring Buffer for Live Scrubbing** — in-memory short-term history for instant time-reversal.
8. **Visualization Filters** — Markov blanket overlay, self-model highlight, physics-law indicators.
9. **WebSocket Optimization** — delta compression, dual-stream at different rates if bandwidth/GC becomes an issue.
10. **Render Loop Interpolation** — delta-time lerp between Web Worker position snapshots for smooth motion on high-refresh-rate (120+ Hz) displays.