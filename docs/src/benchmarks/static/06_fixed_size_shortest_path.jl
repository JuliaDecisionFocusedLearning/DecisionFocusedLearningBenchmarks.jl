# # Shortest Path
# Find the cheapest path from the top-left to the bottom-right of a grid graph:
# edge costs are unknown and must be predicted from instance features.

using DecisionFocusedLearningBenchmarks
using Plots

b = FixedSizeShortestPathBenchmark()

# ## Observable input
#
# At inference time the decision-maker observes the feature vector `x` and the fixed grid
# structure (source top-left, sink bottom-right):
dataset = generate_dataset(b, 50; seed=0)
sample = first(dataset)
plot_context(b, sample)

# ## A training sample
#
# Each sample is a labeled triple `(x, θ, y)`:
# - `x`: instance feature vector (observable at train and test time)
# - `θ`: true edge costs (training supervision only, hidden at test time)
# - `y`: path indicator vector (`y[e] = 1` if edge `e` is on the optimal path)
#
# Top: feature vector x. Bottom left: edge costs θ. Bottom right: optimal path y (white dots):
plot_sample(b, sample)

# ## Untrained policy

# A DFL policy chains two components: a statistical model predicting edge costs:
model = generate_statistical_model(b)     # linear map: features → predicted edge costs
# and a maximizer finding the shortest path given those costs:
maximizer = generate_maximizer(b)         # Dijkstra shortest path on the grid graph

# A randomly initialized policy predicts arbitrary costs, yielding a near-straight path:
θ_pred = model(sample.x)
plot_sample(b, DataSample(; sample.context..., x=sample.x, θ=θ_pred, y=maximizer(θ_pred)))

# Optimality gap on the dataset (0 = optimal, higher is worse):
compute_gap(b, dataset, model, maximizer)

# ---
# ## Problem Description
#
# A **fixed-size grid shortest path** problem. The graph is a directed acyclic grid of
# size ``(\text{rows} \times \text{cols})``, with edges pointing right and downward.
# Edge costs ``\theta \in \mathbb{R}^E`` are unknown; only a feature vector
# ``x \in \mathbb{R}^p`` is observed. The task is to find the minimum-cost path from
# vertex 1 (top-left) to vertex ``V`` (bottom-right):
# ```math
# y^* = \mathrm{argmin}_{y \in \mathcal{P}} \; \theta^\top y
# ```
# where ``y \in \{0,1\}^E`` indicates selected edges and ``\mathcal{P}`` is the set of
# valid source-to-sink paths.
#
# Data is generated following the process in
# [Mandi et al., 2023](https://arxiv.org/abs/2307.13565).
#
# ## Key Parameters
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `grid_size` | Grid dimensions `(rows, cols)` | `(5, 5)` |
# | `p` | Feature dimension | 5 |
# | `deg` | Polynomial degree for cost generation | 1 |
# | `ν` | Multiplicative noise level (0 = no noise) | 0.0 |
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{Features}]{x \in \mathbb{R}^p}
# \fbox{Linear model}
# \xrightarrow[\text{Predicted costs}]{\theta \in \mathbb{R}^E}
# \fbox{Dijkstra / Bellman-Ford}
# \xrightarrow[\text{Path}]{y \in \{0,1\}^E}
# ```
#
# **Model:** `Chain(Dense(p → E))` — predicts one cost per edge.
#
# **Maximizer:** Dijkstra (default) or Bellman-Ford on negated weights to find the
# longest (maximum-weight) path.
#
# !!! note "Reference"
#     Mandi et al. (2023), Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities.
#     [arXiv:2307.13565](https://arxiv.org/abs/2307.13565)
