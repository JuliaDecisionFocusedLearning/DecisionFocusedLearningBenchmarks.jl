# Maintenance problem with resource constraint

The Maintenance problem with resource constraint is a sequential decision-making benchmark where an agent must repeatedly decide which components to maintain over time. The goal is to minimize total expected cost while accounting for independent degradation of components and limited maintenance capacity.


## Problem Description

### Overview

In this benchmark, a system consists of $N$ identical components, each of which can degrade over $n$ discrete states. State $1$ means that the component is new, state $n$ means that the component is failed. At each time step, the agent can maintain up to $K$ components.  

This forms an endogenous multistage stochastic optimization problem, where the agent must plan maintenance actions over the horizon.

### Mathematical Formulation

The maintenance problem can be formulated as a finite-horizon Markov Decision Process (MDP) with the following components:

**State Space** $\mathcal{S}$: At time step $t$, the state $s_t \in [1:n]^N$ is the degradation state for each component.

**Action Space** $\mathcal{A}$: The action at time $t$ is the set of components that are maintained at time $t$:
```math
a_t \subseteq \{1, 2, \ldots, N\} \text{ such that } |a_t| \leq K
```
### Transition Dynamics

The state transitions depend on whether a component is maintained or not:

For each component \(i\) at time \(t\):

- **Maintained component** (\(i \in a_t\)):

\[
s_{t+1}^i = 1 \quad \text{(perfect maintenance)}
\]

- **Unmaintained component** (\(i \notin a_t\)):

\[
s_{t+1}^i =
\begin{cases}
\min(s_t^i + 1, n) & \text{with probability } p,\\
s_t^i & \text{with probability } 1-p.
\end{cases}
\]

Here, \(p\) is the degradation probability, \(s_t^i\) is the current state of component \(i\), and \(n\) is the maximum (failed) state.

---

### Cost Function

The immediate cost at time \(t\) is:

$$
c(s_t, a_t) = \Big( c_m \cdot |a_t| + c_f \cdot \#\{ i : s_t^i = n \} \Big)
$$

Where:

- $c_m$ is the maintenance cost per component.  
- $|a_t|$ is the number of components maintained.  
- $c_f$ is the failure cost per failed component.  
- $\#\{ i : s_t^i = n \}$ counts the number of components in the failed state.

This formulation captures the total cost for maintaining components and penalizing failures.

**Objective**: Find a policy $\pi: \mathcal{S} \to \mathcal{A}$ that minimizes the expected cumulative cost:
```math
\min_\pi \mathbb{E}\left[\sum_{t=1}^T c(s_t, \pi(s_t)) \right]
```

**Terminal Condition**: The episode terminates after $T$ time steps, with no terminal reward.

## Key Components

### [`MaintenanceBenchmark`](@ref)

The main benchmark configuration with the following parameters:

- `N`: number of components (default: 2)
- `K`: maximum number of components that can be maintained simultaneously (default: 1) 
- `n`: number of degradation states per component (default: 3)
- `p`: degradation probability (default: 0.2)
- `c_f`: failure cost (default: 10.0)
- `c_m`: maintenance cost (default: 3.0)
- `max_steps`: Number of time steps per episode (default: 80)

### Instance Generation

Each problem instance includes:

- **Starting State**: Random starting degradation state in $[1,n]$ for each components.

### Environment Dynamics

The environment tracks:
- Current time step
- Current degradation state.

**State Observation**: Agents observe a normalized feature vector containing the degradation state of each component.

## Benchmark Policies

### Greedy Policy  

Greedy policy that maintains components in the last two degradation states, up to the maintenance capacity. This provides a simple baseline.

