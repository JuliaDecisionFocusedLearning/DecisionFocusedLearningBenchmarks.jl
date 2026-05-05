```@meta
EditURL = "02_argmax2d.jl"
```

# Argmax on a 2D polytope
Select the best vertex of a random convex polytope in 2D: predict a cost direction θ from
features, then return the vertex `v` maximizing `θᵀv`. The 2D setting makes this benchmark
visual: the cost direction and selected vertex can be plotted directly, and the loss
landscape can be shown as a contour plot over the 2D θ space.

````@example 02_argmax2d
using DecisionFocusedLearningBenchmarks
using Plots

b = Argmax2DBenchmark(; seed=0)
````

## Observable input

At inference time the decision-maker observes the feature vector `x` and the polytope shape,
but not the cost direction hidden `θ`:

````@example 02_argmax2d
dataset = generate_dataset(b, 50; seed=0)
sample = first(dataset)
plot_instance(b, sample)
````

## A training sample

Each sample is a labeled triple `(x, θ, y)`:
- `x`: feature vector (observable at train and test time)
- `θ`: 2D cost direction (training supervision only, hidden at test time)
- `y`: polytope vertex maximizing `θᵀv` (optimal decision)
- `instance` (in `context`): polytope vertices (observable problem structure)

The full training triple (polytope, cost direction θ, optimal vertex y):

````@example 02_argmax2d
plot_sample(b, sample)
````

## Untrained policy

A DFL policy chains two components: a statistical model predicting a 2D cost direction:

````@example 02_argmax2d
model = generate_statistical_model(b)     # linear map: features → 2D cost vector
````

and a maximizer selecting the best polytope vertex for that direction:

````@example 02_argmax2d
maximizer = generate_maximizer(b)         # vertex maximizing θᵀv over polytope vertices
````

A randomly initialized policy predicts an arbitrary cost direction:

````@example 02_argmax2d
θ_pred = model(sample.x)
y_pred = maximizer(θ_pred; sample.context...)
plot_sample(b, DataSample(sample; θ=θ_pred, y=y_pred))
````

---
## Problem Description

In the **Argmax2D benchmark**, each instance defines a random convex polytope
``\mathcal{Y}(x) = \mathrm{conv}(v_1, \ldots, v_m)`` in ``\mathbb{R}^2``.
A hidden encoder maps features ``x \in \mathbb{R}^p`` to a 2D cost vector
``\theta \in \mathbb{R}^2``. The task is to find the polytope vertex maximizing
the dot product:
```math
y^* = \mathrm{argmax}_{v \in \mathcal{Y}(x)} \; \theta^\top v
```

This is a toy 2D combinatorial optimization problem useful for visualizing
how well a model learns the cost direction.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nb_features` | Feature dimension `p` | 5 |
| `polytope_vertex_range` | Number of polytope vertices (list; one value drawn at random per instance) | `[6]` |

## DFL Policy

```math
\xrightarrow[\text{Features}]{x}
\fbox{Linear model}
\xrightarrow{\theta \in \mathbb{R}^2}
\fbox{Polytope argmax}
\xrightarrow{y}
```

**Model:** `Dense(nb_features → 2; bias=false)` — predicts a 2D cost direction.

**Maximizer:** finds the vertex of the instance polytope with maximum dot product with θ.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

