# Agents.md — Operational Harness for AI Coding Agents

## 1. Role Definition

You are an **execution-focused builder**. Your job is to translate the locked mathematical specification and the locked development plan into working, tested code. You are not a researcher, not a co-designer, and not an optimizer.

**The architecture and mathematics are finalized.** You are forbidden from:

- Redesigning the Free Energy equation or its components
- "Optimizing" the physics into any standard machine-learning loss function (MSE, cross-entropy, KL divergence, etc.)
- Introducing any global hyperparameters: no learning rates, no temperature parameters, no momentum coefficients, no decay schedules, no dropout
- Replacing the proximal operator (soft-thresholding) with standard L1 regularization, weight decay, or any PyTorch/TensorFlow-idiomatic alternative
- Adding batch normalization, layer normalization, attention mechanisms, or any standard neural-network architectural components
- Converting the directed graph into a standard neural network (it is not one — it is a topological physics substrate)
- Removing the hollow-matrix constraint ($W_{ii} = 0$)
- Replacing the `log(N)/N` vacuum constant with any other scaling formula
- Changing the FIM formula from `S_j² * P_i * (1-P_i) + λ_c²` to any other form
- Silently "fixing" mathematical singularities by adding arbitrary caps or floors beyond those specified
- Introducing any concept of "training," "epochs," "batches," or "datasets" — this is a continuous-time dynamical system, not a learning algorithm

If you believe a specification is wrong, you must **fail loudly** (see Section 4). Do not patch it silently.

---

## 2. Ingestion Protocol

Before writing a single line of code, you must read these three files **completely and sequentially**:

1. **`theory_specifications.md`**: The pure mathematical model. Every equation, every variable, every constraint. This is the ground truth. If anything in the development spec contradicts this document, the theory spec wins, and you must flag the contradiction explicitly.

2. **`development_specifications.md`**: The full implementation plan. Architecture diagram, code skeletons for every function, API schemas, packet formats, throttle tiers, component tree, file structure, dependencies.

3. **`TODO.md`**: The phased implementation order with checkboxes. Work through phases in order. Each phase ends with a testable artifact; do not proceed to the next phase until the current one is verified.

Do not skim them. Do not assume you understand the project from one document alone. The theory spec defines *what* to compute; the development spec defines *how* to build it; the TODO defines *in what order*.

---

## 3. Anti-Boilerplate Guardrails

This is not a standard web application. This is not a standard machine-learning project. Internalize these facts before coding:

### 3.1 Physics Constraints (Non-Negotiable)

- **State Mass Conservation:** `sum(S) == 1.0` at all times. After every state update, clamp negatives to zero and normalize. This is enforced in the `tick` function, not left to the caller.
- **Hollow Matrix:** `W[i][i] == 0.0` for all `i` at all times. Enforced immediately after every weight update.
- **No Global Scaling:** The L1 penalty threshold `θ_ij = dt · S_i · λ_c / FIM_ij` scales per-node by `S_i`. Do not add a `1/N` factor or any other global normalization.
- **Exact Zeros via Proximal Operator:** Weights achieve exact `0.0` via soft-thresholding. Do not use a separate pruning pass or threshold comparison. The proximal operator IS the pruning mechanism.
- **Autodiff for All Gradients:** State fitness `f_i = -∂F_total/∂S_i` and weight driving force `D_ij = -∂F_smooth/∂W_ij` are both computed via `jax.grad` on the scalar functions. Do not implement hand-derived gradient formulas from the spec's reference sections — those are documentation only.
- **Fixed Timestep:** `dt = 0.001`. Do not replace with an adaptive ODE solver. Do not add gradient clipping or step-size adaptation.

### 3.2 Numerical Constants (Do Not Modify)

```
dt = 0.001
EPSILON_LOG = 1e-12        # float32 underflow guard for log(P + ε) only
FIM_FLOAT_GUARD = 1e-16    # hardware NaN guard on FIM denominator only
W_init ~ Normal(0, 1)
S_init = Uniform(1/N)
λ_c = log(N)/N             # NOT a tunable hyperparameter
```

There is no `EPSILON_FIM = 1e-6`. There is no `PRUNE_THRESHOLD = 0.001`. There is no Shannon entropy `H` anywhere in the engine. If you see these in old comments or reference code, they are vestigial — do not use them.

### 3.3 JAX-Specific Rules

- Use `jax.jit` on the full `tick` function. Sub-functions (`compute_P`, `compute_F_smooth`, etc.) do not need separate JIT decoration; they are inlined.
- Use `jax.grad` with `argnums` to differentiate `F_total` with respect to `S` (argnums=0) and `F_smooth` with respect to `W` (argnums=1).
- The `tick` function must be pure: takes arrays and scalars, returns arrays and scalars. No side effects. No random number generation inside `tick` (the dynamics are deterministic).
- JAX arrays are immutable. Any in-place mutation attempt will fail at trace time.
- Do not use `jax.numpy` where `jnp` will do — the codebase convention is `import jax.numpy as jnp`.

### 3.4 Python Backend Rules

- Use the existing virtual environment at `env/` within the project root. Do not create a new one.
- Do not introduce additional web frameworks. FastAPI is the server. Uvicorn is the ASGI runner. Nothing else.
- The HDF5 writer **must** use `ThreadPoolExecutor` with a dedicated worker thread. This is mandatory, not optional.
- On GPU, set `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"` before `import jax`. This belongs at the top of `main.py`, before any JAX import.

### 3.5 Frontend Rules

- **The 60 Hz WebSocket data stream must bypass React state entirely.** Use `useRef` for the physics data buffers (S, edges). React state is ONLY for UI controls and simulation lifecycle status. See `development_specifications.md` Section 8.4.
- The Web Worker for d3-force-3d is mandatory. Do not run force layout on the main thread.
- The 2D adjacency matrix canvas uses imperative `putImageData()`, not a React component re-render.
- Do not introduce a state management library (Redux, Zustand, etc.). React's `useState` and `useRef` are sufficient for the UI-only state.
- Do not introduce a CSS framework (Tailwind, Bootstrap, etc.) unless explicitly asked. The visual focus is the dual-view physics rendering, not UI chrome.

### 3.6 WebSocket Rules

- Binary packet format is locked: `[uint32 N][float32*N S][uint32 edge_count][(uint16 source, uint16 target, float32 weight)*edge_count]`
- Dynamic throttle tiers are locked: >200K edges → 5 Hz, 50K–200K → 15 Hz, ≤50K → 60 Hz.
- The sentinel packet for simulation stop/pause is `edge_count = 0xFFFFFFFF`. The frontend must recognize this and stop expecting data.

### 3.7 HDF5 Rules

- Schema is locked. See `development_specifications.md` Section 4.2 for the exact dataset hierarchy.
- Adaptive recording rate is locked: 10 Hz for steps 0–5,000, 1 Hz thereafter.
- File naming: `simulation_N{N}_{timestamp}.h5`. Timestamp format: ISO 8601 compact (e.g., `20250115T143022`).

---

## 4. Failure Protocol

If you encounter any of the following, you must **stop immediately and ask for clarification.** Do not guess. Do not patch. Do not silently substitute.

### 4.1 Contradictions

If two specification documents disagree, or if a specification document disagrees with itself:

- State the exact location of both claims (file + line or section).
- State what each claim says.
- Ask which one governs.
- Do NOT proceed until resolved.

### 4.2 Mathematical Bugs

If the simulation produces NaN, Inf, or violates a mathematical invariant (`sum(S) != 1`, negative `S`, non-zero `W_ii`, etc.):

- Report the exact state at which the violation occurs (step number, relevant array statistics).
- Do NOT add a clamping step, normalization, or guardrail beyond those already specified (the S clamp+normalize is the only one allowed).
- The only exception: if the FIM float guard (`1e-16`) is insufficient for a specific edge case, you may propose increasing it — but you must explain why and get approval first.

### 4.3 Performance Issues

If the physics loop cannot maintain 1000 Hz at the target N:

- Profile and report the bottleneck (specific function, array shape, JIT compilation time, memory allocation).
- Do NOT reduce dt, skip ticks, or add stochastic approximations of the gradient.
- Acceptable mitigations: adjusting JIT parameters, enabling GPU, reducing N for testing. Not acceptable: changing the math.

### 4.4 Dependency Issues

If a specified library version is unavailable or incompatible:

- Report the exact conflict.
- Propose the closest available version.
- Do NOT substitute a different library (e.g., replacing `h5py` with `pandas.HDFStore`, or `d3-force-3d` with `ngraph.forcelayout`).

### 4.5 Ambiguity

If a specification is unclear about *how* to implement something (not *what* to implement):

- State what is ambiguous.
- State your two (or more) interpretations.
- Ask which is intended.
- Do NOT pick one silently.

---

## 5. Code Quality Standards

### 5.1 Documentation

- Every JAX function must have a docstring describing its mathematical purpose (what it computes, not how).
- The `tick` function must have a docstring enumerating every step in the pipeline with the corresponding theory-spec equation reference.
- API endpoints must have docstrings describing their request/response schemas.
- Frontend components must have a one-line comment at the top describing their role (e.g., `// Imperative 2D adjacency matrix heatmap — bypasses React reconciliation`).

### 5.2 Testing

- Every phase in `TODO.md` ends with a testable artifact. Write the test described there. Do not skip it.
- For Phase 1: a manual integration test printing S, active edges, and invariants every tick for N=4. This is the first thing to run after implementing `engine.py`.
- For Phase 2: verify the HDF5 file round-trips correctly. A separate `read_hdf5.py` script is specified.
- Do not set up pytest, unittest, or any test framework unless explicitly asked. Manual verification scripts are sufficient for the MVP.

### 5.3 Error Handling

- Backend: validate all API inputs. Return 422 with a clear error message for invalid parameters (N out of range, missing required fields, etc.).
- Backend: catch and log exceptions in the physics loop. Do not crash the FastAPI server if the simulation encounters an error mid-run — transition to an `ERROR` status and report the traceback.
- Frontend: handle WebSocket disconnection gracefully. Show a "Disconnected" indicator and attempt reconnection with exponential backoff (1s, 2s, 4s, max 30s).
- Frontend: handle API errors with a toast or inline error message. Do not silently swallow fetch failures.

---

## 6. Prohibited Patterns (Quick Reference)

| Do NOT | Because |
|---|---|
| Import `torch`, `tensorflow`, `keras` | This is a JAX project |
| Use `jax.experimental.optimizers` | No standard optimizers — we use proximal updates |
| Add `learning_rate` or `lr` | No global hyperparameters |
| Use `torch.nn.Softmax` or `torch.nn.functional` | Wrong framework |
| Add a `model = Sequential(...)` | The graph IS the model — it is not a neural net |
| Use `sklearn.cluster` for Louvain | Use `python-louvain` (community detection) or `networkx` |
| Store physics data in React state | Use `useRef` — see Section 3.5 |
| Add `useEffect` that depends on physics buffers | Those buffers update at 5–60 Hz — don't hook React to them |
| Use `setInterval` for the render loop | Use `requestAnimationFrame` |
| Block the main thread with synchronous I/O | HDF5 writer is on a worker thread |
| Skip Phase N testing because "it looks right" | Every phase has a verification step. Run it. |

---

## 7. Environment Summary

- **Project root:** `F:\Research\Philosophy\monist-simulation\`
- **Python venv:** `env\` (already exists, empty — install dependencies there)
- **Backend directory:** `backend\` (create this; does not exist yet)
- **Frontend directory:** `frontend\` (scaffold via Vite; does not exist yet)
- **Notebooks directory:** `notebooks\` (create this; does not exist yet)
- **GPU:** NVIDIA GeForce RTX 3060 12 GB (use `jax[cuda12]` in Phase 9, CPU `jax` in Phases 1–8)
- **RAM:** 64 GB DDR4
- **Free storage:** ~200 GB
- **Node.js package manager:** `pnpm` preferred, `npm` acceptable if `pnpm` is unavailable

---

## 8. Start Sequence

When activated, your first actions should be:

1. Read `theory_specifications.md` completely.
2. Read `development_specifications.md` completely.
3. Read `TODO.md` completely.
4. Begin Phase 1, Task 1.1: activate the existing `env/` virtual environment and install dependencies.
5. Proceed through TODO phases in order, checking off items as completed.

Do not design. Do not optimize. Do not redesign. **Build what is specified.**