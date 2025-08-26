# Dynamic Assortment

The Dynamic Assortment problem is a sequential decision-making benchmark where an agent must repeatedly select which subset of items to offer to customers over time. The goal is to maximize total revenue while accounting for dynamic customer preferences that evolve based on purchase history.

## Problem Description

### Overview

In the dynamic assortment problem, a retailer has access to a catalog of ``N`` items and must decide which subset of exactly ``K`` items to offer to customers at each time step. Customers make purchasing decisions according to a choice model that depends on public features ``x``:

- **Item prices**: Fixed monetary cost of each item
- **Item features**: Static characteristics of each item (size ``d``)
- **Hype**: Dynamic popularity that increases when items are purchased recently, and decays over time if not purchased
- **Saturation**: Dynamic measure that slightly increases when specific items are purchased

Both hype and saturation evolve over time based on the agent's assortment decisions and customer purchases, this providing an endogenous multistage stochastic optimization problem.

### Mathematical Formulation

The dynamic assortment problem can be formulated as a finite-horizon Markov Decision Process (MDP) with the following components:

**State Space** ``\mathcal{S}``: At time step ``t``, the state ``s_t`` consists of:
```math
s_t = (p, f, h_t, \sigma_t, t, \mathcal{H}_t)
```
where:
- ``p \in \mathbb{R}^N`` are the fixed item prices
- ``f \in \mathbb{R}^{d \times N}`` are the static item features
- ``h_t \in \mathbb{R}^N`` are the current hype levels for each item
- ``\sigma_t \in \mathbb{R}^N`` are the current saturation levels for each item
- ``t \in \{1, 2, \ldots, T\}`` is the current time step
- ``\mathcal{H}_t`` is the purchase history (last 5 purchases)

**Action Space** ``\mathcal{A}``: The action at time ``t`` is an assortment selection:
```math
a_t \subseteq \{1, 2, \ldots, N\} \text{ such that } |a_t| = K
```

**Customer Choice Model**: Given assortment ``a_t``, customers choose according to a multinomial logit model:
```math
\forall i\in a_t,\, \mathbb{P}(i | a_t, s_t) = \frac{\exp(\theta_i(s_t))}{\sum_{j\in a_t} \exp(\theta_j(s_t)) + 1}
```
```math
\mathbb{P}(\text{no purchase} | a_t, s_t) = \frac{1}{\sum_{j\in a_t} \exp(\theta_j(s_t)) + 1}
```

where ``\theta_i(s_t)`` is the utility of item ``i`` at state ``s_t``, computed by a hidden utility function:
```math
\theta_i(s_t) = \Phi(p_i, h_t^{(i)}, \sigma_t^{(i)}, f_{\cdot,i})
```

**Transition Dynamics** ``\mathcal{P}(s_{t+1} | s_t, a_t)``: After selecting assortment ``a_t`` and observing customer choice ``i^\star \sim \mathbb{P}(\cdot | a_t, s_t)``, the state evolves as:

1. **Hype Update**: For each item ``i``, compute a hype multiplier based on recent purchase history:
   ```math
   m^{(i)} = 1 + \sum_{k=1}^{\min(5, |\mathcal{H}_t|)} \mathbf{1}_{i = \mathcal{H}_t[-k]} \cdot \alpha_k
   ```
   where ``\mathcal{H}_t[-k]`` is the ``k``-th most recent purchase, and the factors are:
   ```math
   \alpha_1 = 0.02, \quad \alpha_2 = \alpha_3 = \alpha_4 = \alpha_5 = -0.005
   ```
   Then update: ``h_{t+1}^{(i)} = h_t^{(i)} \times m^{(i)}``

2. **Saturation Update**:
   ```math
   \sigma_{t+1}^{(i)} = \begin{cases}
   \sigma_t^{(i)} \times 1.01 & \text{if } i = i^\star \\
   \sigma_t^{(i)} & \text{otherwise}
   \end{cases}
   ```

3. **History Update**: ``\mathcal{H}_{t+1} = \text{append}(\mathcal{H}_t, i^\star)`` (keeping last 5 purchases)

**Reward Function** ``r(s_t, a_t, s_{t+1})``: The immediate reward is the revenue from the customer's purchase:
```math
r(s_t, a_t, s_{t+1}) = \begin{cases}
p_{i^\star} & \text{if customer purchases item } i^\star \\
0 & \text{if no purchase}
\end{cases}
```

**Objective**: Find a policy ``\pi: \mathcal{S} \to \mathcal{A}`` that maximizes the expected cumulative reward:
```math
\max_\pi \mathbb{E}\left[\sum_{t=1}^T r(s_t, \pi(s_t), s_{t+1}) \right]
```

**Terminal Condition**: The episode terminates after ``T`` time steps, with no terminal reward.

## Key Components

### [`DynamicAssortmentBenchmark`](@ref)

The main benchmark configuration with the following parameters:

- `N`: Number of items in the catalog (default: 20)
- `d`: Dimension of static feature vectors (default: 2) 
- `K`: Assortment size constraint (default: 4)
- `max_steps`: Number of time steps per episode (default: 80)
- `customer_choice_model`: linear mapping from features to utilities
- `exogenous`: Whether dynamics are exogenous (default: false)

### Instance Generation

Each problem instance includes:

- **Prices**: Random values in [1, 10] for each item, plus 0 for no-purchase
- **Features**: Random static features in [1, 10] for each item
- **Initial State**: Random starting hype and saturation values in [1, 10]

### Environment Dynamics

The environment tracks:
- Current time step
- Purchase history (last 5 purchases)
- Current hype and saturation for each item  
- Customer utilities computed from current state

**State Observation**: Agents observe a normalized feature vector containing:
- Current full features (prices, hype, saturation, static features)
- Change in hype/saturation from previous step
- Change in hype/saturation from initial state  
- Normalized current time step

All features are divided by 10 for normalization.

## Benchmark Policies

### Expert Policy

The expert policy computes the optimal assortment by brute-force enumeration:
1. Enumerate all possible K-subsets of the N items
2. For each subset, compute expected revenue using the choice model
3. Return the subset with highest expected revenue

This provides an optimal baseline but is computationally expensive.

### Greedy Policy  

The greedy policy selects the K items with the highest prices, ignoring dynamic effects and customer preferences. This provides a simple baseline.

## Decision-Focused Learning Policy

```math
\xrightarrow[\text{State}]{s_t}
\fbox{Neural network $\varphi_w$}
\xrightarrow[\text{Cost vector}]{\theta}
\fbox{Top K}
\xrightarrow[\text{Assortment}]{a_t}
```

**Components**:

1. **Neural Network** ``\varphi_w``: Takes the current state ``s_t`` as input and predicts item utilities ``\theta = (\theta_1, \ldots, \theta_N)``
2. **Optimization Layer**: Selects the top ``K`` items with highest predicted utilities to form the assortment ``a_t``

## Reference

Based on the paper: [Structured Reinforcement Learning for Combinatorial Decision-Making](https://arxiv.org/abs/2505.19053)
