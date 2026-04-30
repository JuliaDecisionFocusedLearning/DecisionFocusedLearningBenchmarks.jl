# # Subset Selection
# Select the `k` most valuable items from a set of `n`: items with unknown values
# must be identified from observable features alone.

using DecisionFocusedLearningBenchmarks
using Plots

b = SubsetSelectionBenchmark(; identity_mapping=false)

# ## Observable input
#
# At inference time the decision-maker observes only the feature vector `x`:
dataset = generate_dataset(b, 50; seed=0)
sample = first(dataset)
plot_instance(b, sample)

# ## A training sample
#
# Each sample is a labeled triple `(x, θ, y)`:
# - `x`: item feature vector (observable at train and test time)
# - `θ`: true item values, derived from `x` via a hidden encoder (training supervision only)
# - `y`: selection indicator (`y[i] = 1` for the `k` highest-value items, 0 otherwise)
#
# The full training triple (features, hidden values, and selection):
plot_sample(b, sample)

# ## Untrained policy

# A DFL policy chains two components: a statistical model predicting item scores:
model = generate_statistical_model(b)     # linear map: features → predicted item scores
# and a maximizer selecting the top-k items by those scores:
maximizer = generate_maximizer(b)         # top-k selection

# A randomly initialized policy selects items with no relation to their true values:
θ_pred = model(sample.x)
plot_sample(b, DataSample(; sample.context..., x=sample.x, θ=θ_pred, y=maximizer(θ_pred)))

# Optimality gap on the dataset (0 = optimal, higher is worse):
compute_gap(b, dataset, model, maximizer)

# ---
# ## Problem Description
#
# In the **Subset Selection benchmark**, ``n`` items have unknown values ``\theta_i``.
# A feature vector ``x \in \mathbb{R}^n`` is observed (identity mapping by default).
# The task is to select the ``k`` items with the highest values:
# ```math
# y = \mathrm{top}_k(\theta)
# ```
# where ``y \in \{0,1\}^n`` with exactly ``k`` ones.
#
# ## Key Parameters
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `n` | Total number of items | 25 |
# | `k` | Number of items to select | 5 |
# | `identity_mapping` | Use identity as the hidden mapping | `true` |
#
# When `identity_mapping=true`, features equal item values directly (`x = θ`).
# When `false`, a random linear layer is used as the hidden mapping.
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{Features}]{x}
# \fbox{Linear model}
# \xrightarrow{\theta}
# \fbox{top-k}
# \xrightarrow{y}
# ```
#
# **Model:** `Dense(n → n; bias=false)` — predicts a score per item.
#
# **Maximizer:** `top_k(θ, k)` — returns a boolean vector with `true` at the `k`
# highest-scoring positions.
#
# !!! note "Reference"
#     Setting from [Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities](https://arxiv.org/abs/2307.13565)
