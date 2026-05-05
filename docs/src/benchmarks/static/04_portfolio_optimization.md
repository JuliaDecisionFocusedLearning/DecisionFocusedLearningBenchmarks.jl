```@meta
EditURL = "04_portfolio_optimization.jl"
```

# Portfolio Optimization
Allocate wealth across assets to maximize expected return subject to a risk constraint:
asset returns are unknown and must be predicted from contextual features.

````@example 04_portfolio_optimization
using DecisionFocusedLearningBenchmarks
using Plots

b = PortfolioOptimizationBenchmark()
````

## Observable input

At inference time the decision-maker observes only the contextual feature vector `x`:

````@example 04_portfolio_optimization
dataset = generate_dataset(b, 20; seed=0)
sample = first(dataset)
plot_instance(b, sample)
````

## A training sample

Each sample is a labeled triple `(x, θ, y)`:
- `x`: contextual feature vector (observable at train and test time)
- `θ`: true expected asset returns (training supervision only, hidden at test time)
- `y`: optimal portfolio weights solving the Markowitz QP given `θ`

Top: feature vector x. Bottom left: true returns θ. Bottom right: optimal weights y:

````@example 04_portfolio_optimization
plot_sample(b, sample)
````

## Untrained policy

A DFL policy chains two components: a statistical model predicting expected asset returns:

````@example 04_portfolio_optimization
model = generate_statistical_model(b)     # linear map: features → predicted returns
````

and a maximizer allocating the optimal portfolio given those returns:

````@example 04_portfolio_optimization
maximizer = generate_maximizer(b)         # Markowitz QP solver (Ipopt via JuMP)
````

A randomly initialized policy predicts arbitrary returns, leading to a suboptimal allocation:

````@example 04_portfolio_optimization
θ_pred = model(sample.x)
plot_sample(b, DataSample(; sample.context..., x=sample.x, θ=θ_pred, y=maximizer(θ_pred)))
````

Optimality gap on the dataset (0 = optimal, higher is worse):

````@example 04_portfolio_optimization
compute_gap(b, dataset, model, maximizer)
````

---
## Problem Description

A **Markowitz portfolio optimization** problem where asset expected returns are unknown.
Given contextual features ``x \in \mathbb{R}^p``, the learner predicts returns
``\theta \in \mathbb{R}^d`` and solves:

```math
\begin{aligned}
\max_{y} \quad & \theta^\top y \\
\text{s.t.} \quad & y^\top \Sigma y \leq \gamma \\
& \mathbf{1}^\top y \leq 1 \\
& y \geq 0
\end{aligned}
```

where ``\Sigma`` is the asset covariance matrix and ``\gamma`` is the risk budget.
The solver uses [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) via JuMP.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `d` | Number of assets | 50 |
| `p` | Feature dimension | 5 |
| `deg` | Polynomial degree for data generation | 1 |
| `ν` | Noise hyperparameter | 1.0 |

Data is generated following the process in
[Mandi et al., 2023](https://arxiv.org/abs/2307.13565).

## DFL Policy

```math
\xrightarrow[\text{Features}]{x \in \mathbb{R}^p}
\fbox{Linear model}
\xrightarrow[\text{Predicted returns}]{\hat{\theta} \in \mathbb{R}^d}
\fbox{QP solver (Ipopt)}
\xrightarrow[\text{Portfolio}]{y \in \mathbb{R}^d}
```

**Model:** `Dense(p → d)` — predicts one expected return per asset.

**Maximizer:** Ipopt QP solver enforcing the variance and budget constraints.

!!! note "Reference"
    Mandi et al. (2023), Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities.
    [arXiv:2307.13565](https://arxiv.org/abs/2307.13565)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

