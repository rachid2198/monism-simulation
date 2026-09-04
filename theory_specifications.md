# Theory Specification: Project Prismatic Monism

## 1. Philosophical Foundation

Project Prismatic Monism models the universe as a single directed graph whose topology and internal state evolution are driven entirely by information-theoretic principles. The universe is a monist mental structure that learns itself, and in the process naturally dissociates into semi-autonomous subsystems (alters) — represented here as Markov-blanket-like subgraphs.

### Core Tenets

- **Monism:** Matter (state probabilities, S) and spacetime (topological weights, W) are two attributes of a single unified directed graph.
- **Self-Modeling:** The universe learns about itself by modeling and predicting its own internal states. Subjective experience resides in the model of the Markov blanket, not the blanket itself.
- **Zero Static Hyper-parameters:** No global learning rates, static time-delays, or manual tuning dials exist. Temporal separation and structural plasticity emerge dynamically from Information Geometry.

---

## 2. Fundamental Variables

### Network Dimensions
- $N$: Total number of nodes in the graph (Integer, e.g., $N = 4$ for testing, $N = 1000$ for full simulation).

### Core Data Structures

**State Vector $S$:**
- Type: $\mathbb{R}^N$, Float32
- Constraint: $\sum_{i} S_i = 1.0$ and $0 \leq S_i \leq 1$ for all $i$.
- Interpretation: The distribution of "attention mass" or "energy" across the graph. Represents how much of the universe's total state mass is localized at each fundamental coordinate.

**Topological Weight Matrix $W$:**
- Type: $\mathbb{R}^{N \times N}$, Float32
- Constraint: $W_{ii} = 0$ (hollow matrix — no self-connections). Row $i$ = target node, Column $j$ = source node.
- Special Value: $W_{ij} = 0.0$ represents a severed connection (topological boundary). Edges achieve exact zero via proximal soft-thresholding.

**Generative Prediction Vector $P$:**
- Type: $\mathbb{R}^N$, Float32
- Constraint: $\sum_k P_k = 1.0$, $P_k > 0$ for all $k$.
- Interpretation: The network's internal prediction of the state distribution.

---

## 3. Derived Quantities

### 3.1 Generative Prediction (Softmax)
Given states $S$ and weights $W$, the network predicts its own next state:

$$z_i = \sum_j W_{ij} \cdot S_j$$

$$P_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$$

$P$ is a global function of $S$ and $W$; changing any $S_j$ or $W_{ij}$ affects every $P_k$.

### 3.2 Vacuum Constant (Erdős–Rényi Connectivity Threshold)
The cosmological vacuum energy of the universe, representing the asymptotic phase-transition threshold for maximum-entropy random graphs:

$$\lambda_c = \frac{\ln(N)}{N}$$

This constant serves as the universe's native L1 regularization scale — the baseline "static friction" or metabolic cost of maintaining a connection.

### 3.3 Fisher Information Metric (Total)
The true total local curvature of the information manifold, combining the curvature from predictive likelihood and the curvature from the vacuum's Laplace prior:

$$F^{\text{Total}}_{ij} = S_j^2 \cdot P_i \cdot (1 - P_i) + \lambda_c^2$$

- $S_j^2 \cdot P_i \cdot (1 - P_i)$: Fisher Information of the categorical likelihood (data curvature).
- $\lambda_c^2$: Fisher Information of the Laplace prior with rate parameter $\lambda_c$ (vacuum curvature). This follows from the fact that an L1 penalty is equivalent to a Laplace(0, $1/\lambda_c$) prior whose Fisher Information is exactly $\lambda_c^2$.

The total FIM is a diagonal approximation: each weight $W_{ij}$ receives its own local curvature. A microscopic float guard (e.g., $10^{-16}$) is applied purely to prevent hardware underflow; it is not a physics parameter.

### 3.4 Free Energy (Localized per Node)
The scalar cost function that the universe minimizes. This is the pure thermodynamic form:

$$F_i = S_i \cdot \left( -\ln(P_i + \epsilon_{\log}) + \lambda_c \sum_j |W_{ij}| \right)$$

- $-\ln(P_i + \epsilon_{\log})$: **Prediction error (accuracy).** Low when $P_i$ is high — the node's state matches the network's prediction.
- $\lambda_c \sum_j |W_{ij}|$: **Complexity cost (structure).** Proportional to the total L1 mass of the node's outgoing edges, weighted by the vacuum constant.
- $S_i$: **State mass weighting.** A node with zero state mass pays zero metabolic cost; a node with high state mass pays proportionally. This is a local, not global, constraint — the penalty does not scale with $N$.

Where $\epsilon_{\log}$ (e.g., $10^{-12}$) is a pure float32 underflow guard, not a physics parameter.

The total free energy of the universe is:

$$F_{\text{total}} = \sum_i F_i$$

---

## 4. Dynamics

### 4.1 Fast Dynamics: State Evolution (Replicator Dynamics)
States flow along the probability simplex according to the natural gradient of free energy:

**Step 1: Compute the raw Euclidean gradient (fitness).**

$$f_i = -\frac{\partial F_{\text{total}}}{\partial S_i}$$

This gradient is computed via automatic differentiation and includes all softmax cross-derivatives (chain rule through $P$, through $|W_{ij}|$, and through the $S_i$ prefactor in every $F_k$).

**Step 2: Project onto the simplex via Replicator Dynamics.**

$$\bar{f} = \sum_k S_k \cdot f_k$$

$$\frac{dS_i}{dt} = S_i \cdot (f_i - \bar{f})$$

This is the natural gradient on the probability simplex: state mass flows toward nodes with above-average fitness and away from nodes with below-average fitness, while automatically preserving $\sum S_i = 1$.

**Step 3: Numerical integration (Explicit Euler with Guardrails).**

$$S_i^{\text{next}} = S_i + \Delta t \cdot \frac{dS_i}{dt}$$

$$S_i^{\text{clamped}} = \max(0, S_i^{\text{next}})$$

$$S_i^{\text{new}} = \frac{S_i^{\text{clamped}}}{\sum_k S_k^{\text{clamped}}}$$

The fixed integration timestep $\Delta t = 0.001$ serves as the fundamental "Planck tick" of the simulation.

### 4.2 Slow Dynamics: Weight Evolution (Natural Gradient + Proximal Operator)

**Step 1: Compute the smooth predictive driving force.**

Define the smooth component of the total free energy (no L1 terms — those are handled analytically by the proximal operator):

$$F_{\text{smooth}} = -\sum_i S_i \cdot \ln(P_i + \epsilon_{\log})$$

$$D_{ij}^{\text{driving}} = -\frac{\partial F_{\text{smooth}}}{\partial W_{ij}}$$

This driving force is computed via automatic differentiation. It represents the "predictive gravity" — how much changing $W_{ij}$ reduces prediction error.

**Step 2: Take a natural gradient step (temporary update).**

$$W_{ij}^{\text{temp}} = W_{ij} + \Delta t \cdot \frac{D_{ij}^{\text{driving}}}{F^{\text{Total}}_{ij}}$$

**Step 3: Apply the Proximal Operator (Soft-Thresholding).**

The L1 penalty $\lambda_c \sum_j |W_{ij}|$ is handled exactly via the proximal map for the Laplace prior under the Fisher metric:

$$\theta_{ij} = \frac{\Delta t \cdot S_i \cdot \lambda_c}{F^{\text{Total}}_{ij}}$$

$$W_{ij}^{\text{new}} = \text{sign}(W_{ij}^{\text{temp}}) \cdot \max(0, \, |W_{ij}^{\text{temp}}| - \theta_{ij})$$

**Step 4: Enforce Hollow Matrix Constraint.**

$$W_{ii}^{\text{new}} = 0 \quad \forall i$$

---

### 4.3 Interpretation of the Dynamics

- **Predictive Gravity ($D^{\text{driving}}$):** Pulls weights in directions that improve the network's predictions. Edges that help explain active states strengthen.
- **Vacuum Tax ($\theta_{ij}$):** The L1 proximal threshold acts as a "static friction" that must be overcome for an edge to survive. Below the threshold, an edge snaps to exactly zero. Above it, the edge survives but pays a constant metabolic cost.
- **Regrowth:** A severed edge ($W_{ij} = 0$) can regrow if the predictive driving force later exceeds the vacuum tax — the proximal operator permits regrowth naturally.
- **State Mass Coupling:** Both the driving force and the threshold are scaled by the local state mass $S_i$, so node regions with low state mass have frozen (inert) topology.

---

## 5. Initial Conditions ($t = 0$)

**States:** Uniform distribution.
$$S_i = \frac{1}{N} \quad \forall i$$

**Weights:** IID from standard normal.
$$W_{ij} \sim \mathcal{N}(0, 1) \quad \forall i \neq j$$
$$W_{ii} = 0 \quad \forall i$$

**Constants:**
- $\Delta t = 0.001$ (fundamental integration timestep)
- $\epsilon_{\log} = 10^{-12}$ (float32 underflow guard for logarithm)
- $\epsilon_{\text{FIM}} = 10^{-16}$ (hardware underflow guard for FIM denominator)
- $\lambda_c = \frac{\ln(N)}{N}$ (vacuum constant)

---

## 6. Summary of Key Equations

| Quantity | Formula |
|---|---|
| Prediction | $P_i = \text{softmax}(W \cdot S)_i$ |
| Vacuum Constant | $\lambda_c = \ln(N) / N$ |
| Total FIM (per edge) | $F^{\text{Total}}_{ij} = S_j^2 \cdot P_i \cdot (1 - P_i) + \lambda_c^2$ |
| Node Free Energy | $F_i = S_i \cdot (-\ln(P_i + \epsilon_{\log}) + \lambda_c \sum_j \vert W_{ij}\vert)$ |
| Total Free Energy | $F_{\text{total}} = \sum_i F_i$ |
| State Fitness | $f_i = -\partial F_{\text{total}} / \partial S_i$ |
| State Derivative | $\dot{S_i} = S_i \cdot (f_i - \sum_k S_k f_k)$ |
| Smooth Free Energy | $F_{\text{smooth}} = -\sum_i S_i \cdot \ln(P_i + \epsilon_{\log})$ |
| Weight Driving Force | $D_{ij} = -\partial F_{\text{smooth}} / \partial W_{ij}$ |
| Natural Gradient Step | $W_{ij}^{\text{temp}} = W_{ij} + \Delta t \cdot D_{ij} / F^{\text{Total}}_{ij}$ |
| Proximal Threshold | $\theta_{ij} = \Delta t \cdot S_i \cdot \lambda_c / F^{\text{Total}}_{ij}$ |
| Proximal Update | $W_{ij}^{\text{new}} = \text{sign}(W_{ij}^{\text{temp}}) \cdot \max(0, \vert W_{ij}^{\text{temp}}\vert - \theta_{ij})$ |
| Hollow Constraint | $W_{ii} = 0$ |
| State Guardrails | Clamp to $\geq 0$, normalize to sum $= 1$ |