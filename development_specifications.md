# Development Specification: Project Prismatic Monism

## 1. Architecture Overview

Monolithic Python backend: JAX physics engine → HDF5 recording → FastAPI WebSocket server. Frontend: Vite + React + Three.js with a Web Worker for force-directed layout.

```
┌─────────────────────────────────────────────────────────┐
│  Backend (Python)                                       │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ JAX Loop │───▶│ HDF5     │    │ FastAPI          │   │
│  │ 1000 Hz  │    │ Snapshot │    │ /start /pause    │   │
│  │          │    │ Writer   │    │ /resume /reset   │   │
│  │          │    │          │    │ /analyze         │   │
│  └──────────┘    └──────────┘    │ WS ws://...:8000 │   │
│                                  └────────┬─────────┘   │
└───────────────────────────────────────────┼─────────────┘
                                            │ 60 Hz (throttled)
┌───────────────────────────────────────────┼─────────────┐
│  Frontend (Vite + React + Three.js)       │             │
│                                           ▼             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐      │
│  │ React UI │    │ Web Worker   │    │ Three.js │      │
│  │ Controls │    │ Force Layout │───▶│ 3D Scene │      │
│  │ State    │    │ (d3-force-3d)│    │          │      │
│  └──────────┘    └──────────────┘    └──────────┘      │
│                                           │             │
│  ┌──────────────────────────────────────┐ │             │
│  │ 2D Canvas: Adjacency Matrix Heatmap  │◀┘             │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Physics Engine (JAX)

### 2.1 Function Definitions

All computation is JIT-compiled. The core tick function signature:

```python
@jax.jit
def physics_tick(S, W, dt, lambda_c, eps_log):
    # Returns: S_new, W_new, diagnostics
```

**Sub-functions (all JIT-inlined):**

```python
def compute_P(S, W):
    z = W @ S
    return jax.nn.softmax(z)

def compute_F_smooth(S, W, eps_log):
    P = compute_P(S, W)
    return -jnp.sum(S * jnp.log(P + eps_log))

def compute_F_total(S, W, lambda_c, eps_log):
    P = compute_P(S, W)
    log_term = -jnp.log(P + eps_log)
    l1_term = lambda_c * jnp.sum(jnp.abs(W), axis=1)
    F_per_node = S * (log_term + l1_term)
    return jnp.sum(F_per_node), F_per_node

def compute_FIM(S, W, P, lambda_c):
    # Per-edge Fisher Information
    fim = (S[None, :] ** 2) * P[:, None] * (1.0 - P[:, None]) + lambda_c**2
    fim = jnp.maximum(fim, 1e-16)  # hardware guard
    return fim

def replicator_step(S, f):
    f_mean = jnp.sum(S * f)
    dS_dt = S * (f - f_mean)
    return dS_dt

def proximal_weight_update(W, driving_force, FIM, S, dt, lambda_c):
    # Natural gradient step
    W_temp = W + dt * driving_force / FIM
    # Proximal soft-threshold
    threshold = dt * S[:, None] * lambda_c / FIM
    W_new = jnp.sign(W_temp) * jnp.maximum(0.0, jnp.abs(W_temp) - threshold)
    # Hollow matrix
    W_new = W_new * (1.0 - jnp.eye(W.shape[0]))
    return W_new
```

### 2.2 Per-Tick Pipeline (JIT-compiled)

```python
def tick(S, W, dt, lambda_c, eps_log):
    P = compute_P(S, W)
    
    # State update (autodiff on F_total)
    F_total, F_per_node = compute_F_total(S, W, lambda_c, eps_log)
    f = -jax.grad(lambda s: compute_F_total(s, W, lambda_c, eps_log)[0])(S)
    dS_dt = replicator_step(S, f)
    S_new = S + dt * dS_dt
    S_new = jnp.maximum(S_new, 0.0)
    S_new = S_new / jnp.sum(S_new)
    
    # Weight update (autodiff on F_smooth + proximal)
    F_smooth = compute_F_smooth(S, W, eps_log)
    driving_force = -jax.grad(compute_F_smooth, argnums=1)(S, W, eps_log)
    FIM = compute_FIM(S, W, P, lambda_c)
    W_new = proximal_weight_update(W, driving_force, FIM, S, dt, lambda_c)
    
    active_edges = jnp.count_nonzero(W_new)
    delta_W = jnp.mean(jnp.abs(W_new - W))
    delta_S = jnp.mean(jnp.abs(S_new - S))
    
    return S_new, W_new, active_edges, delta_W, delta_S
```

### 2.3 Integration Parameters

- `dt = 0.001` — fixed timestep (Explicit Euler)
- Physics loop runs at 1000 Hz (1000 ticks per second of simulation time)
- `EPSILON_LOG = 1e-12` — float32 underflow guard in `log(P + eps)`
- `FIM_FLOAT_GUARD = 1e-16` — hardware NaN guard on total FIM denominator

---

## 3. Initialization

### 3.1 State and Weight Setup

```python
def initialize(N, seed):
    key = jax.random.PRNGKey(seed)
    
    # Uniform state distribution
    S = jnp.ones(N) / N
    
    # Standard normal weights
    W = jax.random.normal(key, (N, N))
    W = W * (1.0 - jnp.eye(N))  # hollow matrix
    
    lambda_c = jnp.log(N) / N
    
    return S, W, lambda_c
```

### 3.2 Configuration

- `N`: node count (4–1000), configurable at startup
- `seed`: RNG seed, user-provided or random
- `max_steps`: optional safety cap (default 100,000)
- `dt`, `eps_log`: fixed constants

---

## 4. HDF5 Recording

### 4.1 Adaptive Recording Rate

| Timesteps | Frequency | Interval |
|---|---|---|
| 0 – 5,000 | 10 Hz | Every 100th step |
| > 5,000 | 1 Hz | Every 1000th step |

### 4.2 HDF5 File Schema

```
/sim_N              (scalar uint16)   — node count
/sim_seed           (scalar uint32)   — random seed
/sim_dt             (scalar float32)  — integration timestep
/sim_lambda_c       (scalar float32)  — vacuum constant
/sim_max_steps      (scalar uint32)   — maximum step count configured
/snapshots/step     (1D uint32, M)    — timestep index per snapshot
/snapshots/S        (2D float32, M×N) — state vector per snapshot
/snapshots/W_i      (1D uint16, K)    — row indices (sparse)
/snapshots/W_j      (1D uint16, K)    — column indices (sparse)
/snapshots/W_val    (1D float32, K)   — weight values (sparse)
/snapshots/W_ptr    (1D uint32, M+1)  — pointer into W_* arrays
/snapshots/active_edges (1D uint32, M) — edge count per snapshot
/snapshots/delta_W  (1D float32, M)   — mean weight change
/snapshots/delta_S  (1D float32, M)   — mean state change
```

- `M` = total snapshots written (varies with adaptive rate)
- `K` = total non-zero edges across all snapshots
- `W_ptr[t]` to `W_ptr[t+1]` gives the slice for snapshot `t`
- File naming convention: `simulation_N{N}_{timestamp}.h5`

### 4.3 Writer Implementation (Mandatory Async)

- Use `h5py` library
- Write once per snapshot tick (not every physics tick)
- **The HDF5 writer must run on a dedicated worker thread via `ThreadPoolExecutor` with a single worker and an unbounded queue.** Snapshot data (S, W arrays) is copied and enqueued on the main thread; the worker thread serializes to HDF5. This eliminates any I/O jitter from the 1000 Hz physics loop, regardless of disk load.
- On simulation end, drain the queue and join the worker before closing the file.

---

## 5. Thermodynamic Auto-Pause (Equilibrium Detection)

### 5.1 Sliding Window

- Window size: 500 ticks (0.5 seconds of simulation time)
- Track two rolling means: $\overline{\Delta W}$ and $\overline{\Delta S}$
- Where $\Delta W = \text{mean}_{i,j} |W_{ij}^{\text{new}} - W_{ij}|$ and $\Delta S = \text{mean}_i |S_i^{\text{new}} - S_i|$

### 5.2 Convergence Threshold

- If both $\overline{\Delta W} < 10^{-6}$ AND $\overline{\Delta S} < 10^{-6}$ for 500 consecutive ticks, the simulation auto-pauses
- Status set to `PAUSED_EQUILIBRIUM`
- A notification is sent to the frontend via WebSocket

---

## 6. Simulation Lifecycle

### 6.1 State Machine

```
                  start()
    IDLE ─────────────────────▶ RUNNING
     ▲                           │  │
     │                    pause()│  │resume()
     │              equilibrium  │  │
     │              auto-pause   ▼  │
     │                    PAUSED◀──┘
     │                       │
     └───────────────────────┘
            reset()
```

### 6.2 REST Endpoints

| Method | Path | Body | Response | Description |
|---|---|---|---|---|
| POST | `/simulation/start` | `{N, seed, max_steps?}` | `200 {status: "running"}` | Initialize and start |
| POST | `/simulation/pause` | — | `200 {status: "paused", step}` | Pause the loop |
| POST | `/simulation/resume` | — | `200 {status: "running"}` | Resume from paused |
| POST | `/simulation/reset` | — | `200 {status: "idle"}` | Reset to IDLE |
| GET | `/simulation/status` | — | `{status, step, N, seed, ...}` | Current state |
| POST | `/simulation/analyze` | `{step, analysis_type}` | `{clusters, spectral_coords, ...}` | Deep analysis |

### 6.3 Lifecycle State

```python
class SimStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    PAUSED_EQUILIBRIUM = "paused_equilibrium"  # auto-pause
    STOPPED_MAX_STEPS = "stopped_max_steps"      # safety cap
```

---

## 7. WebSocket Streaming

### 7.1 Binary Packet Format

```
[Header]
  uint32: N (node count) — 4 bytes

[State Vector]
  float32[N]: S (state probabilities) — 4*N bytes

[Edge Data]
  uint32: edge_count — 4 bytes
  For each edge (edge_count entries):
    uint16: source_idx (j) — 2 bytes
    uint16: target_idx (i) — 2 bytes
    float32: weight (W_ij) — 4 bytes
    Total per edge: 8 bytes
```

Total packet size: `8 + 4*N + 8*edge_count` bytes.

### 7.2 Dynamic Throttle

| Tier | Edge Count | WS Broadcast Rate | Rationale |
|---|---|---|---|
| Plasma | > 200,000 | 5 Hz | ~8 MB/packet; V8 GC safe. Web Worker lerps to 60 Hz. |
| Transition | 50,000 – 200,000 | 15 Hz | ~400 KB–1.6 MB/packet. Lerp 4:1 to 60 Hz. |
| Crystalline | ≤ 50,000 | 60 Hz | ≤ 400 KB/packet. GC trivial. No lerp needed. |

The server re-evaluates the tier each physics tick. The WebSocket send loop checks whether enough real time has passed since the last send for the current tier.

### 7.3 Frontend WebSocket Handling

- WebSocket `onmessage` fires at the throttled rate (5/15/60 Hz)
- **Do not store incoming arrays in React state.** Use `useRef` to imperatively update:
  - `sRef.current`: Float32Array of state (for Canvas 2D and Three.js)
  - `edgesRef.current`: typed arrays of edge source/target/weight (for Canvas 2D, Three.js, and Web Worker)
- The Web Worker receives a `postMessage` with the edge data whenever new data arrives
- The Web Worker runs d3-force-3d at its own pace, posting interpolated node positions back to the main thread at 60 Hz via `requestAnimationFrame`-aligned messages

---

## 8. Frontend Architecture

### 8.1 Technology Stack

- **Build:** Vite
- **Framework:** React 18+
- **3D Rendering:** Three.js (via npm, not CDN)
- **Force Layout:** d3-force-3d (runs in a Web Worker)
- **2D Canvas:** Native HTML5 Canvas API (via `useRef`)

### 8.2 Component Tree

```
<App>
  ├── <ControlPanel>
  │     ├── N selector (slider or input, 4–1000)
  │     ├── Seed input + "Random" checkbox
  │     ├── Max Steps input
  │     ├── Start / Pause / Resume / Reset buttons
  │     ├── Status indicator (idle / running / paused / equilibrium)
  │     └── "Pause & Analyze" button (enabled when paused)
  │
  ├── <AdjacencyMatrix>
  │     └── <canvas> (2D, rows/columns sorted by S_i descending)
  │
  ├── <ThreeScene>
  │     └── Three.js WebGL renderer (nodes as InstancedMesh spheres, edges as BufferGeometry lines)
  │
  └── <AnalysisPanel> (shown after "Pause & Analyze")
        ├── Cluster assignment table/list
        └── Spectral embedding visualization (optional 3D overlay)
```

### 8.3 React State (UI Only)

```typescript
interface UIState {
  N: number;
  seed: number | null;
  useRandomSeed: boolean;
  maxSteps: number;
  simStatus: 'idle' | 'running' | 'paused' | 'paused_equilibrium' | 'stopped_max_steps';
  currentStep: number;
  activeEdges: number;
  analysisResult: AnalysisResult | null;
  analysisLoading: boolean;
}
```

### 8.4 Non-React Buffers (useRef)

```typescript
// Updated imperatively from WebSocket onmessage
const sBufferRef = useRef<Float32Array | null>(null);
const edgeSourceRef = useRef<Uint16Array | null>(null);
const edgeTargetRef = useRef<Uint16Array | null>(null);
const edgeWeightRef = useRef<Float32Array | null>(null);
```

### 8.5 Web Worker: Force-Directed Layout

- Library: d3-force-3d
- Forces:
  - `forceLink`: weighted by `|W_ij|` as spring stiffness
  - `forceManyBody`: charge-based repulsion (-strength)
  - `forceCenter`: weak gravity toward origin
- The worker receives new edge/weight data whenever the WebSocket fires
- When data arrives, the worker updates the simulation's links and restarts the alpha (cooling parameter)
- The worker runs `simulation.tick(n)` in batches, posting node positions `[x0,y0,z0, x1,y1,z1, ...]` to the main thread at ~60 Hz
- Between data arrivals (e.g., during 5 Hz plasma phase), the layout simulates freely using the last received topology, keeping the 3D view smooth

### 8.6 2D Adjacency Matrix Canvas

- Sorted by current `S_i` descending (re-sort on each data arrival)
- Pixel (i, j) color = `|W[row_i][col_j]|` mapped to a color scale (white=0, hot=strong)
- Canvas size: e.g., 600×600 pixels. For N > 600, each cell is 1 pixel; for N < 600, cells are scaled up
- Imperatively drawn via `CanvasRenderingContext2D.putImageData()` for performance

---

## 9. Pause & Analyze

### 9.1 Flow

1. User clicks "Pause & Analyze"
2. If simulation is running, frontend sends `POST /simulation/pause`
3. Frontend sends `POST /simulation/analyze` with `{step: currentStep, analysis_type: "louvain"}`
4. Backend reads the HDF5 snapshot for that step
5. Backend runs Louvain community detection on `|W|` (undirected, weighted by absolute weight magnitude)
6. Backend optionally runs spectral embedding (first 3 non-trivial Laplacian eigenvectors) for cluster visualization
7. Backend returns JSON: `{clusters: [{id, nodes: [int], size}], spectral_coords: float[][] | null}`
8. Frontend re-sorts the adjacency matrix by cluster assignment (instead of S_i), coloring cluster boundaries
9. Frontend optionally animates the 3D graph from its force-directed positions to spectral coordinates

### 9.2 Analysis Types (extensible)

| Type | Key | Description |
|---|---|---|
| Louvain | `"louvain"` | Community detection via modularity maximization |
| Spectral | `"spectral"` | Laplacian eigendecomposition for intrinsic geometry |
| Markov Blanket | `"blanket"` | Placeholder — deferred feature |
| Self-Model | `"self_model"` | Placeholder — deferred feature |

### 9.3 Backend Analysis Implementation

- Use `scipy.sparse.csgraph` for Laplacian
- Use `scipy.sparse.linalg.eigsh` for spectral
- Use `networkx` + `community-louvain` (or a JAX-native implementation) for Louvain
- Analysis runs synchronously on the FastAPI event loop (acceptable since the simulation is paused)

---

## 10. Three.js Rendering Details

### 10.1 Nodes

- `THREE.InstancedMesh` with `SphereGeometry(radius, 16, 16)`
- Radius per instance = `baseRadius + scale * S_i`
- Color/emissive per instance = gradient based on S_i (low=cool blue, high=hot white)
- Positions updated from Web Worker's Float32Array each frame

### 10.2 Edges

- `THREE.BufferGeometry` with `LineSegments`
- Each edge is one line segment: two vertices (source pos, target pos)
- Opacity = `min(1.0, |W_ij| * opacity_scale)`
- Color = white/light gray
- Geometry rebuilt when edge count changes; positions updated each frame from Web Worker
- For very high edge counts in plasma phase, render only a random subset (e.g., 50,000) to maintain 60 FPS

### 10.3 Render Loop

- `requestAnimationFrame` drives the Three.js renderer
- Each frame: read latest node positions from the shared Float32Array, update InstancedMesh matrix and edge BufferGeometry positions, render
- The Canvas 2D adjacency matrix updates at the data arrival rate (5/15/60 Hz), not at render rate (60 Hz)

---

## 11. File Structure

```
monist-simulation/
├── theory_specifications.md       # Pure mathematical model (reference)
├── development_specifications.md  # This file (implementation plan)
├── TODO.md                        # Step-by-step implementation order
├── requirements.txt               # Single Python dependency manifest (all packages)
├── backend/
│   ├── main.py                    # FastAPI app entry point, REST routes, WebSocket
│   ├── engine.py                  # JAX physics engine (all JIT functions)
│   ├── lifecycle.py               # Simulation state machine, run loop
│   ├── recorder.py                # HDF5 snapshot writer
│   ├── websocket_handler.py       # WebSocket manager, throttling
│   └── analysis.py                # Pause-time analysis (Louvain, spectral)
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx               # React entry
│   │   ├── App.tsx                # Root component, layout
│   │   ├── components/
│   │   │   ├── ControlPanel.tsx
│   │   │   ├── AdjacencyMatrix.tsx
│   │   │   ├── ThreeScene.tsx
│   │   │   └── AnalysisPanel.tsx
│   │   ├── hooks/
│   │   │   ├── useSimulation.ts   # REST API calls
│   │   │   └── useWebSocket.ts    # WebSocket connection + buffer updates
│   │   ├── workers/
│   │   │   └── forceLayout.worker.ts  # Web Worker for d3-force-3d
│   │   └── utils/
│   │       ├── packetParser.ts    # Binary packet decode
│   │       └── colorScales.ts     # Color mapping utilities
│   └── public/
└── notebooks/
    └── analysis_demo.ipynb        # Example: loading HDF5, plotting, etc.
```

---

## 12. Dependencies

All Python dependencies are declared in a single root `requirements.txt`. Install with `pip install -r requirements.txt`. Frontend dependencies live in `frontend/package.json` and are installed via `pnpm install`.

### Backend (Python)

```
jax[cpu]          # or jax[cuda12] for GPU
fastapi
uvicorn[standard]
h5py
numpy
scipy
networkx
python-louvain    # community detection
websockets        # (stdlib, or via fastapi)
```

For GPU: `jax[cuda12]` + CUDA 12 + cuDNN.

**GPU Memory Configuration:** Set the environment variable `XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` before importing JAX. This reserves ~7.2 GB for JAX on the 12 GB RTX 3060, leaving ~4.8 GB free for the browser's WebGL contexts (Three.js + Canvas). JAX defaults to preallocating 75% of VRAM, which would leave insufficient headroom. This must be set in the Python process startup (e.g., `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"` in `main.py` before any `import jax`).

### Frontend (Node.js)

```
three
d3-force-3d
react
react-dom
@types/react
@types/react-dom
@types/three
typescript
vite
```

---

## 13. Future Work / Deferred Features

1. **Markov Blanket Detection:** Pause-time analysis to identify conditional-independence boundaries in the graph. Exact algorithm TBD (mutual information + greedy partition).
2. **Self-Model Discovery:** Detect subgraphs whose internal $P$ distribution mirrors the global $S$ distribution — the hypothesized seat of subjective experience.
3. **Physics-Isomorphism Checks:** Analyze whether the network's emergent dynamics replicate physical laws (thermodynamics, quantum mechanics, relativity analogs).
4. **Timeline Scrubber:** UI control for scrubbing through saved HDF5 snapshots with replay visualization.
5. **Spectral Clustering:** Alternative to Louvain for the "Pause & Analyze" view. Backend eigendecomposition is straightforward; frontend rendering of spectral coordinates is deferred.
6. **Jupyter Notebooks:** Offline deep-dive analysis notebooks for loading HDF5 and running custom metrics.
7. **Ring Buffer for Live Scrubbing:** If live time-reversal is needed without re-reading HDF5.
8. **Additional Visualization Filters:** Markov blanket overlay, self-model highlight, physics-law compliance indicators.
9. **Delta Compression / Dual-Stream Optimization:** If WebSocket bandwidth or GC pressure becomes an issue at scale.
10. **Render Loop Interpolation:** On high-refresh-rate displays (120+ Hz), the 60 Hz Web Worker position stream causes duplicate frames. Add delta-time-based lerp interpolation in the Three.js render loop between the two most recent worker position snapshots for buttery motion on all display refresh rates.