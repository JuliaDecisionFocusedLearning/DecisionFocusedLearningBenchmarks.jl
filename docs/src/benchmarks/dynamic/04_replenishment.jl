# # Dynamic Replenishment
# A retailer must decide how many units of each item to reorder at each time step. Demand
# follows an endogenous customer choice model, in which the purchase probability depends on
# the items currently available and on the stock levels.
# The objective is to maximize total revenue over a finite horizon, defined as the sales
# margin minus the stock costs.
# Replenished items take a certain amount of time to reach the store, but they can already be
# sold while in transit. The physical inventory has soft lower and upper bounds, and violating
# them incurs a large penalty.
# Items are also subject to coupling production constraints with random quotas.

using DecisionFocusedLearningBenchmarks
using Plots

b = DynamicReplenishmentBenchmark()

# ## Observable input
#
# Generate one environment and roll it out with the random policy to collect a sample
# trajectory. At each step the agent observes item prices and features, the current virtual
# and physical stock levels, the sales, replenishment and stock histories, and the remaining
# quotas:
policies = generate_baseline_policies(b)
env = generate_environments(b, 1)[1]
_, trajectory = evaluate_policy!(policies.random, env)

# The observable state at step 1: stock levels (virtual stock includes the units still in
# transit, physical stock only the units already in the store) together with the static
# utility of each item, which drives the customer choice model:
plot_context(b, trajectory[1])

# ## A training sample
#
# Each step in a trajectory is a labeled tuple `(x, (θ, η), y)` plus state and reward:
# - `x`: `(d+19) × ∑ᵢ ubᵢ` feature matrix per step, with one column per candidate stock level
#   of each item (`ubᵢ` is the replenishment upper bound of item ``i``). The rows hold the
#   static features (price and item features), the dynamic item features (current stock, mean
#   sales, mean stock, mean number of customers, days on lot) and the stock-level features
#   (deviations from the stock bounds, the mean stock and the quota). The last row is the item
#   identifier.
# - `(θ, η)`: predicted utility scores. `θ` is the predicted utility of each item, `η` the
#   predicted marginal cost of each additional unit held in stock.
# - `y`: replenishment decision at this step (vector of length ``N``)
# - `instance`: the state, containing the physical and virtual stock levels together with the
#   sales, replenishment and stock histories
# - `reward`: sales margin minus stock costs at time step ``t``
#
# One step in the trajectory:
plot_sample(b, trajectory[1])

# A few steps side by side:
plot_trajectory(b, trajectory[1:min(4, length(trajectory))])

# ## DFL pipeline components

# The DFL agent chains two components: a neural network predicting utility scores per item:
model = generate_statistical_model(b)     # state features → predicted utility scores (θ, η)
# and a maximizer choosing the best replenishment decision based on the predicted scores,
# the stock levels and the production constraints:
maximizer = generate_maximizer(b)

# At each step, the model maps the current state (prices, features, stock levels, quotas) to
# the utility scores ``(\theta, \eta)``. The maximizer then selects the feasible replenishment
# decision that maximizes the total predicted utility.

# ---
# ## Problem Description
#
# ### Overview
#
# In the **Dynamic Replenishment problem**, a retailer has a catalog of ``N`` items, of which
# only a subset is present in its inventory at any given time. The inventory splits into units
# that are physically in the store and units that are still in transit; the items offered to
# customers are all those with a positive virtual or physical inventory.
# The retailer pays a holding cost for every unit in stock and for every unit in transit, and
# it faces soft lower and upper bounds on the physical stock level: violating either bound
# incurs a penalty.
# At each time step the retailer decides how many units of each item to replenish, subject to
# coupling production quotas that limit how many units can be reordered.
# Customer demand is stochastic and follows a multinomial logit choice model.
#
# The problem is characterized by:
# - **Endogenous noise**: what customers buy depends on which items are actually available, hence on the past replenishment decisions
# - **Combinatorial action space**: the number of feasible replenishment decisions is exponential in the number of items
#
# ### Mathematical Formulation
#
# **State** ``s_t = (p, f, vs_t, ps_t, t, \mathcal{H}_t^s, \mathcal{H}_t^r, \mathcal{H}_t^p, \mathcal{H}_t^c)`` where:
# - ``p``: fixed item prices
# - ``f``: static item features
# - ``vs_t``: current virtual stock levels
# - ``ps_t``: current physical stock levels
# - ``\mathcal{H}_t^s``: stock history 
# - ``\mathcal{H}_t^r``: revenue history 
# - ``\mathcal{H}_t^p``: purchase history 
# - ``\mathcal{H}_t^c``: customer history 
# - ``t``: current time step
#
# **Action:** ``a_t \in \mathbb{N}^N`` with ``A a_t \leq b_t``, where ``A`` is the (coupling) production constraint matrix and ``b_t`` the quotas of each constraint at time step ``t``.
#
# **Customer choice** (multinomial logit): each item is assigned a static utility score ``v_i``
# given by the customer choice model. The default model is linear, so the utility of an item is
# a linear combination of its features. Only the items that are actually in the inventory can
# be bought, that is the offer set ``\mathcal{O}_t = \{i : vs_t^i > 0\}``. The probability that
# a customer purchases item ``i`` at time step ``t`` is:
# ```math
# \mathbb{P}(i \mid s_t) = \frac{\exp(v_i)}{\sum_{j \in \mathcal{O}_t} \exp(v_j) + 1}
# ```
# where the ``+1`` in the denominator accounts for the no-purchase option.
# The Gumbel-max trick is used to sample the purchased item from this distribution: customer
# ``k`` purchases the item ``i^\star`` such that
# ```math
# i^\star = \operatorname*{argmax}_{i \in \mathcal{O}_t \cup \{0\}} \left(v_i + \epsilon_i^k\right)
# ```
# where ``\epsilon_i^k`` is a Gumbel random variable and ``0`` denotes the no-purchase option.
# At each time step, a random number of customers (drawn from a Poisson distribution of rate
# ``\lambda``) arrives and makes purchases. We write ``q_t^i`` for the number of units of item
# ``i`` purchased at time step ``t``.
#
# **Transition dynamics:** for each item ``i``, with ``\tau`` the delivery delay:
# - ``vs_{t+1}^i = vs_t^i + a_t^i - q_t^i``
# - ``ps_{t+1}^i = \max(0, ps_t^i + a_{t-\tau}^i - q_t^i)``
#
# **Reward:** for each item ``i``:
# - the sales margin is ``q_t^i m_i``
# - the virtual stock cost is ``c_{vs}^i \, vs_t^i``
# - the physical stock cost is ``c_{ps}^i \, ps_t^i``
# - the penalty for violating the soft stock bounds is ``c_{lb}^i \max(0, lb_i - ps_t^i) + c_{ub}^i \max(0, ps_t^i - ub_i)``
#
# The total reward at time step ``t`` is therefore:
# ```math
# r(s_t, a_t) = \sum_{i=1}^N q_t^i m_i - c_{vs}^i \, vs_t^i - c_{ps}^i \, ps_t^i - c_{lb}^i \max(0, lb_i - ps_t^i) - c_{ub}^i \max(0, ps_t^i - ub_i)
# ```
#
# **Objective:**
# ```math
# \max_\pi \; \mathbb{E}\!\left[\sum_{t=1}^T r(s_t, \pi(s_t))\right]
# ```
#
# ## Key Components
#
# ### [`DynamicReplenishmentBenchmark`](@ref)
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `N` | Number of items in the catalog | 10 |
# | `λ` | Poisson arrival rate of customers per step | 15 |
# | `d` | Static feature dimension per item (in addition to price) | 5 |
# | `nb_constraints` | Number of coupling production constraints | 2 |
# | `constraints_matrix` | Coupling matrix ``A`` (`nb_constraints × N`) | random 0/1 matrix |
# | `quotas` | Quotas ``b_t`` per constraint and per step (`max_steps × nb_constraints`) | random in ``[10, 30]`` |
# | `stock_inf` | Soft lower bound on the physical stock | 0 |
# | `stock_sup` | Soft upper bound on the physical stock | 30 |
# | `ub_same_item` | Upper bound on the number of units of the same item | 30 |
# | `delivery_delay` | Delivery delay ``\tau``, in time steps | 3 |
# | `max_steps` | Steps per episode | 10 |
# | `customer_choice_model` | Model mapping item features to static utilities | random linear model |
#
# Prices are drawn uniformly in ``[1, 10]`` and item features uniformly in ``[-10, 10]``. The
# stock costs derive from the prices (``c_{vs}^i = p_i / 10T`` and ``c_{ps}^i = p_i / 5T``),
# and the bound violation cost is ``\max_i p_i``. The static utilities are obtained by applying
# the customer choice model to the standardized features, with a ``0`` appended for the
# no-purchase option.
#
# Either provide only `nb_constraints`, in which case a random constraints matrix and random
# quotas are generated, or provide both `constraints_matrix` and `quotas` and they are used
# as is.
#
# ### State Observation
#
# Agents observe a ``(d+19) \times \sum_i ub_i`` feature matrix, with one column per candidate
# stock level of each item. Each column concatenates:
# - the ``d+1`` static features of the item (price and item features)
# - 9 dynamic item features: current stock, mean sales, mean stock, mean number of past customers, and mean days on lot (each also scaled by the price)
# - 8 stock-level features: the deviations from `stock_inf`, from `stock_sup`, from the mean stock and from the quota of the step (each also scaled by the price)
# - the item identifier, used by the statistical model to group the columns per item
#
# ### Environment Generation
#
# Each environment starts from an initial stock drawn uniformly in ``\{0,\ldots,5\}`` per item,
# from which the virtual and physical stocks, the histories and the per-item replenishment
# upper bounds are initialized. A scenario is sampled at the same time and fixes, for the whole
# episode, the number of customers arriving at each step (Poisson with rate ``\lambda``) and
# the Gumbel perturbation of the utilities of each customer. Resetting the environment restores
# the initial stock and, by default, resamples the scenario.

# ## Baseline Policies
#
# | Policy | Description |
# |--------|-------------|
# | Greedy | Solves the replenishment problem with the prices as item utilities and no stock penalization, which favors the most expensive items |
# | Random | Goes through the items in a random order and replenishes a random feasible quantity of each |
# | Lazy | Never replenishes anything |
# | SAA | Solves a multi-stage sample average approximation of the problem over the remaining horizon, on a set of sampled scenarios, and applies the first-stage decision |
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{State}]{s_t}
# \fbox{Neural network $\varphi_w$}
# \xrightarrow[\text{Utilities}]{(\theta, \eta) \in \mathbb{R}^N \times \mathbb{R}^{\sum_i ub_i}}
# \fbox{Replenishment maximizer}
# \xrightarrow[\text{Replenishment}]{a_t}
# ```
#
# **Model:** two heads sharing the same feature matrix:
# - `θ_model = Chain(Dense(d+10 => 1))`: one replenishment utility ``\theta_i`` per item, from the item features only
# - `η_model = Chain(Dense(d+18 => 1), softplus)`: one nonnegative marginal stock cost ``\eta_{i,j}`` per candidate stock level ``j`` of item ``i``
#
# **Maximizer:** the predicted scores parametrize the linear objective of a MILP solved at each
# step, where ``y_i`` is the replenishment of item ``i``, ``s_i`` its current stock and
# ``z_{i,j} = 1`` if the post-replenishment stock of item ``i`` reaches at least ``j`` units:
# ```math
# \begin{aligned}
# \max_{y, z} \quad & \sum_{i = 1}^N \overbrace{\theta_i y_i}^{\text{replenishment revenue}}
#   + \overbrace{\underbrace{\eta_{i,1} z_{i,1}}_{\text{intercept}}
#   - \sum_{j = 2}^{ub_i} \underbrace{z_{i,j} \sum_{k = 2}^{j} \eta_{i,k}}_{\text{decreasing slope}}}^{\text{stock cost}} \\
# \text{s.t.} \quad
#   & y_i + s_i = \sum_{j = 1}^{ub_i} z_{i,j}, \quad \forall i \in [N] \\
#   & z_{i,j} \geq z_{i,j+1}, \quad \forall i \in [N], \ j \in [ub_i - 1] \\
#   & A y \leq b_t \\
#   & y_i \in \{0, \ldots, ub_i\}, \quad z_{i,j} \in \{0, 1\}
# \end{aligned}
# ```
# The first constraint links the ``z`` variables to the post-replenishment stock, the second
# enforces that they are non-increasing in ``j`` (so they encode a stock level rather than an
# arbitrary set), and the third is the coupling quota constraint of the current step. Since
# ``\eta \geq 0``, the stock cost is a concave piecewise-linear function of the stock level,
# which lets the model penalize large inventories without making the problem nonlinear.
#
