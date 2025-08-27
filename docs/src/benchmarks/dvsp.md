# Dynamic Vehicle Scheduling

The Dynamic Vehicle Scheduling Problem (DVSP) is a sequential decision-making problem where an agent must dynamically dispatch vehicles to serve customers that arrive over time.

## Problem Description

### Overview

In the dynamic vehicle scheduling problem, a fleet operator must decide at each time step which customer requests to serve immediately and which to postpone to future time steps.
The goal is to serve all customers by the end of the planning horizon while minimizing total travel time.

This is a simplified version of the more complex Dynamic Vehicle Routing Problem with Time Windows (DVRPTW), focusing on the core sequential decision-making aspects without capacity or time window constraints.

The problem is characterized by:
- **Exogenous noise**: customer arrivals are stochastic and follow a fixed known distribution, independent of the agent's actions
- **Combinatorial action space**: at each time step, the agent must build vehicle routes to serve selected customers, which leads to a huge combinatorial action space

### Mathematical Formulation

The dynamic vehicle scheduling problem can be formulated as a finite-horizon Markov Decision Process (MDP):

**State Space** ``\mathcal{S}``: At time step ``t``, the state ``s_t`` consists of:
```math
s_t = (R_t, D_t, t)
```
where:
- ``R_t`` are the pending customer requests (not yet served), where each request ``r_i \in R_t`` contains:
  - ``x_i, y_i``: 2d spatial coordinates of the customer location
  - ``\tau_i``: start time when the customer needs to be served
  - ``s_i``: service time required to serve the customer
- ``D_t`` indicates which requests must be dispatched this time step (i.e. that cannot be postponed further, otherwise they will be infeasible at the next time step because of their start time)
- ``t \in \{1, 2, \ldots, T\}`` is the current time step

The state also implicitly includes (constant over time):
- Travel duration matrix ``d_{ij}``: time to travel from location ``i`` to location ``j``
- Depot location

**Action Space** ``\mathcal{A}``: The action at time step ``t`` is a set of vehicle routes:
```math
a_t = \{r_1, r_2, \ldots, r_k\}
```
where each route ``r_i`` is a sequence of customer that starts and ends at the depot.

A route is feasible if:
- It starts and ends at the depot
- It follows time constraints, i.e. customers are served on time

**Transition Dynamics** ``\mathcal{P}(s_{t+1} | s_t, a_t)``: After executing routes ``a_t``:

1. **Remove served customers** from the pending request set
2. **Generate new customer arrivals** according to the underlying exogenous distribution
3. **Update must-dispatch set** based on postponement rules

**Reward Function** ``r(s_t, a_t)``: The immediate reward is the negative total travel time of the routes:

```math
r(s_t, a_t) = - \sum_{r \in a_t} \sum_{(i,j) \in r} d_{ij}
```

where ``d_{ij}`` is the travel duration from location ``i`` to location ``j``, and the sum is over all consecutive location pairs in each route ``r``.

**Objective**: Find a policy ``\pi: \mathcal{S} \to \mathcal{A}`` that maximizes expected cumulative reward:
```math
\max_\pi \mathbb{E}\left[\sum_{t=1}^T r(s_t, \pi(s_t)) \right]
```

## Key Components

### [`DynamicVehicleSchedulingBenchmark`](@ref)

The main benchmark configuration with the following parameters:

- `max_requests_per_epoch`: Maximum number of new customer requests per time step (default: 10)
- `Δ_dispatch`: Time delay between decision and vehicle dispatch (default: 1.0)
- `epoch_duration`: Duration of each decision time step (default: 1.0)
- `two_dimensional_features`: Whether to use simplified 2D features instead of full feature set (default: false)

### Instance Generation

Problem instances are generated from static vehicle routing datasets and include:

- **Customer locations**: Spatial coordinates for pickup/delivery points
- **Depot location**: Central starting and ending point for all routes
- **Travel times**: Distance/duration matrix between all location pairs
- **Service requirements**: Time needed to serve each customer

The dynamic version samples new customer arrivals from the static instance, drawing new customers by independently sampling their locations and service times.

### Features

The benchmark provides two feature representations:

**Full Features** (14-dimensional):
- Start times for postponable requests
- End times (start + service time)
- Travel time from depot to request
- Travel time from request to depot  
- Slack time until next time step
- Quantile-based travel times to other requests (9 quantiles)

**2D Features** (simplified):
- Travel time from depot to request
- Mean travel time to other requests

## Benchmark Policies

### Lazy Policy

The lazy policy postpones all possible requests, serving only those that must be dispatched.

### Greedy Policy  

The greedy policy serves all pending requests as soon as they arrive, without considering future consequences. 

## Decision-Focused Learning Policy

```math
\xrightarrow[\text{State}]{s_t}
\fbox{Neural network $\varphi_w$}
\xrightarrow[\text{Priorities}]{\theta}
\fbox{Prize-collecting VSP}
\xrightarrow[\text{Routes}]{a_t}
```

**Components**:

1. **Neural Network** ``\varphi_w``: Takes current state features as input and predicts customer priorities ``\theta = (\theta_1, \ldots, \theta_n)``
2. **Optimization Layer**: Solves the prize-collecting vehicle scheduling problem to determine optimal routes given the predicted priorities

The neural network architecture adapts to the feature dimensionality:
- **2D features**: `Dense(2 => 1)` followed by vectorization
- **Full features**: `Dense(14 => 1)` followed by vectorization
