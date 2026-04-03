# # Dynamic Vehicle Scheduling
# Dispatch vehicles to customers arriving over time: at each step the agent decides which
# customers to serve now and which to postpone, minimizing total travel cost.

using DecisionFocusedLearningBenchmarks
using Plots

b = DynamicVehicleSchedulingBenchmark()

# ## A sample episode
#
# Generate one environment and roll it out with the greedy policy (serves all pending
# customers immediately):
policies = generate_baseline_policies(b)
env = generate_environments(b, 1)[1]
_, trajectory = evaluate_policy!(policies.greedy, env)

# One step: depot (green square), must-dispatch customers (red stars; deadline reached),
# postponable customers (blue triangles), vehicle routes (lines):
plot_solution(b, trajectory[1])

# Multiple steps side by side — customers accumulate and routes change over time:
plot_trajectory(b, trajectory[1:min(3, length(trajectory))])

# ## DFL pipeline components

# The DFL agent chains two components: a neural network predicting a prize per customer:
model = generate_statistical_model(b)     # Dense(27 → 1) per customer: state features → prize
# and a maximizer selecting routes that balance collected prizes against travel costs:
maximizer = generate_maximizer(b)         # prize-collecting VSP solver

# At each step, the model assigns a prize to each postponable customer. The solver then
# selects routes maximizing collected prizes minus travel costs, deciding which customers
# to serve now and which to defer.

# ---
# ## Problem Description
#
# ### Overview
#
# In the **Dynamic Vehicle Scheduling Problem (DVSP)**, a fleet operator must decide at
# each time step which customers to serve immediately and which to postpone. The goal is
# to serve all customers by end of the planning horizon while minimizing total travel time.
#
# The problem is characterized by:
# - **Exogenous noise**: customer arrivals are stochastic and follow a fixed distribution
# - **Combinatorial action space**: routes are built over a large set of customers
#
# ### Mathematical Formulation
#
# **State** ``s_t = (R_t, D_t, t)`` where:
# - ``R_t``: pending customers, each with coordinates, start time, service time
# - ``D_t``: must-dispatch customers (cannot be postponed further)
# - ``t``: current time step
#
# **Action** ``a_t``: a set of vehicle routes ``\{r_1, r_2, \ldots, r_k\}``, each starting
# and ending at the depot, satisfying time constraints.
#
# **Reward:**
# ```math
# r(s_t, a_t) = -\sum_{r \in a_t} \sum_{(i,j) \in r} d_{ij}
# ```
#
# **Objective:**
# ```math
# \max_\pi \; \mathbb{E}\!\left[\sum_{t=1}^T r(s_t, \pi(s_t))\right]
# ```
#
# ## Key Components
#
# ### [`DynamicVehicleSchedulingBenchmark`](@ref)
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `max_requests_per_epoch` | Maximum new customers per time step | 10 |
# | `Δ_dispatch` | Time delay between decision and dispatch | 1.0 |
# | `epoch_duration` | Duration of each time step | 1.0 |
# | `two_dimensional_features` | Use 2D instead of full 27D features | `false` |
#
# ### Features
#
# **Full features (27D per customer):** start/end times, depot travel times, slack,
# reachability ratios, quantile-based travel times to other customers.
#
# **2D features:** travel time from depot + mean travel time to others.
#
# ## Baseline Policies
#
# | Policy | Description |
# |--------|-------------|
# | Lazy | Postpones all possible customers; serves only must-dispatch |
# | Greedy | Serves all pending customers immediately |
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{State}]{s_t}
# \fbox{Neural network $\varphi_w$}
# \xrightarrow[\text{Prizes}]{\theta}
# \fbox{Prize-collecting VSP}
# \xrightarrow[\text{Routes}]{a_t}
# ```
#
# The neural network predicts a prize ``\theta_i`` for each postponable customer.
# The prize-collecting VSP solver then maximizes collected prizes minus travel costs:
# ```math
# \max_{a_t \in \mathcal{A}(s_t)} \sum_{r \in a_t} \left(\sum_{i \in r} \theta_i - \sum_{(i,j) \in r} d_{ij}\right)
# ```
#
# **Model:**
# - 2D features: `Dense(2 → 1)` applied independently per customer
# - Full features: `Dense(27 → 1)` applied independently per customer
#
# !!! note "Reference"
#     TODO: add original reference.
