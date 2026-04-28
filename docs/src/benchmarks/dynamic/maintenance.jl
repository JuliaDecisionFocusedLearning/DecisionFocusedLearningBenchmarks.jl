# # Maintenance
# Decide which components to maintain at each step to minimize failure and maintenance costs:
# components degrade stochastically and the agent has limited maintenance capacity.

using DecisionFocusedLearningBenchmarks
using Plots

b = MaintenanceBenchmark(; N=5, K=2)  # 5 components, maintain up to 2 per step

# ## A sample episode
#
# Generate one environment and roll out with the greedy policy (maintains the most degraded
# components up to capacity):
policies = generate_baseline_policies(b)
env = generate_environments(b, 1)[1]
_, trajectory = evaluate_policy!(policies.greedy, env)

# One step: bars show degradation levels (1 = new, n = failed), green = maintained, red = failed:
plot_solution(b, trajectory[1])

# A few steps side by side showing degradation evolving over time:
plot_trajectory(b, trajectory[1:min(4, length(trajectory))])

# ## DFL pipeline components

# The DFL agent chains two components: a neural network predicting urgency scores per component:
model = generate_statistical_model(b)     # two-layer MLP: degradation state → urgency scores
# and a maximizer selecting the most urgent components for maintenance:
maximizer = generate_maximizer(b)         # top-K selection among components with positive scores

# At each step, the model maps the current degradation state to an urgency score per component.
# The maximizer selects up to K components with the highest positive scores for maintenance.

# ---
# ## Problem Description
#
# ### Overview
#
# In the **Maintenance benchmark**, a system has ``N`` identical components, each with
# ``n`` discrete degradation states (1 = new, ``n`` = failed). At each step, the agent
# can maintain up to ``K`` components. Maintained components are reset to state 1.
# Unmaintained components degrade stochastically.
#
# ### Mathematical Formulation
#
# **State** ``s_t \in \{1,\ldots,n\}^N``: degradation level of each component.
#
# **Action** ``a_t \subseteq \{1,\ldots,N\}`` with ``|a_t| \leq K``
#
# **Transition dynamics:** For each component ``i``:
# - If maintained: ``s_{t+1}^i = 1``
# - If not maintained: ``s_{t+1}^i = \min(s_t^i + 1, n)`` with probability ``p``, else ``s_t^i``
#
# **Cost:**
# ```math
# c(s_t, a_t) = c_m \cdot |a_t| + c_f \cdot \#\{i : s_t^i = n\}
# ```
#
# **Objective:**
# ```math
# \min_\pi \; \mathbb{E}\!\left[\sum_{t=1}^T c(s_t, \pi(s_t))\right]
# ```
#
# ## Key Components
#
# ### [`MaintenanceBenchmark`](@ref)
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `N` | Number of components | 2 |
# | `K` | Max simultaneous maintenance operations | 1 |
# | `n` | Degradation levels per component | 3 |
# | `p` | Degradation probability per step | 0.2 |
# | `c_f` | Failure cost per failed component | 10.0 |
# | `c_m` | Maintenance cost per maintained component | 3.0 |
# | `max_steps` | Steps per episode | 80 |
#
# ### Instance Generation
#
# Each instance has random starting degradation states uniformly drawn from ``\{1,\ldots,n\}``.
#
# ## Baseline Policies
#
# | Policy | Description |
# |--------|-------------|
# | Greedy | Maintains components in the last degradation state before failure, up to capacity |
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{State}]{s_t \in \{1,\ldots,n\}^N}
# \fbox{Neural network $\varphi_w$}
# \xrightarrow[\text{Scores}]{\theta \in \mathbb{R}^N}
# \fbox{Top-K (positive)}
# \xrightarrow[\text{Maintenance}]{a_t}
# ```
#
# **Model:** `Chain(Dense(N → N), Dense(N → N), vec)` — two-layer MLP predicting one
# urgency score per component.
#
# **Maximizer:** `TopKPositiveMaximizer(K)` — selects the ``K`` components with the
# highest positive scores for maintenance.
