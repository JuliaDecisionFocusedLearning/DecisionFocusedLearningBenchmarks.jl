# # Contextual Stochastic Argmax
# Select the best item from a set of `n` items with stochastic utilities: each scenario draws
# a different utility vector, but utilities depend on observable context features. This is a
# toy benchmark designed so that a linear model can exactly recover the optimal
# context-to-utility mapping.

using DecisionFocusedLearningBenchmarks
using Plots

b = ContextualStochasticArgmaxBenchmark()

# `generate_dataset` returns unlabeled samples (`y = nothing`) for this benchmark.
# A `target_policy` must be provided to attach labels. Here we use the anticipative
# oracle: it returns the item with the highest realized utility for each scenario,
# giving one labeled sample per scenario per instance.
anticipative = generate_anticipative_solver(b)
policy =
    (ctx, scenarios) -> [
        DataSample(; ctx.context..., x=ctx.x, y=anticipative(ξ), extra=(; scenario=ξ))
        for ξ in scenarios
    ]
dataset = generate_dataset(b, 20; target_policy=policy, seed=0)
sample = first(dataset)

# ## Observable input
#
# At inference time the model observes `x = [c_base; x_raw]`. `plot_instance` shows both
# components: base utilities `c_base` (left) and context features `x_raw` (right):
plot_instance(b, sample)

# ## A training sample
#
# Stochastic benchmarks have no single ground-truth label: the optimal item depends on
# which utility scenario is realized. We label each sample with the anticipative oracle,
# which returns the best item given the realized scenario ξ.
#
# Each labeled sample contains:
# - `x`: feature vector `[c_base; x_raw]` (observable at train and test time)
# - `y`: optimal item for the realized scenario ξ (one-hot; anticipative oracle label)
# - `extra.scenario`: realized utility vector ξ (available only during training)
#
# Top: feature vector x. Bottom: realized scenario ξ acting as the cost vector,
# with the anticipative-optimal item in red:
plot_sample(b, DataSample(sample; θ=sample.scenario))

# ## Untrained policy

# A DFL policy chains two components: a statistical model predicting expected item utilities:
model = generate_statistical_model(b)     # linear map: features → predicted expected utilities
# and a maximizer selecting the item with the highest predicted utility:
maximizer = generate_maximizer(b)         # one-hot argmax

# A randomly initialized policy selects items with no relation to their expected utilities.
# Top: feature vector x. Bottom: predicted utilities θ̂ with the selected item in red:
θ_pred = model(sample.x)
plot_sample(b, DataSample(sample; θ=θ_pred, y=maximizer(θ_pred)))

# ---
# ## Problem Description
#
# ### Overview
#
# In the **Contextual Stochastic Argmax benchmark**, ``n`` items have random utilities
# that depend on observable context. Per instance:
# - ``c_\text{base} \sim U[0,1]^n``: base utilities (stored in `context`)
# - ``x_\text{raw} \sim \mathcal{N}(0, I_d)``: observable context features
# - Full features: ``x = [c_\text{base}; x_\text{raw}] \in \mathbb{R}^{n+d}``
#
# The realized utility (scenario) is drawn as:
# ```math
# \xi = c_\text{base} + W \, x_\text{raw} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
# ```
# where ``W \in \mathbb{R}^{n \times d}`` is a fixed unknown perturbation matrix.
#
# The task is to select the item with the highest realized utility:
# ```math
# y^* = \mathrm{argmax}(\xi)
# ```
#
# A linear model ``\theta = [I \mid W] \cdot x`` can exactly recover the optimal
# solution in expectation.
#
# ## Key Parameters
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `n` | Number of items | 10 |
# | `d` | Context feature dimension | 5 |
# | `noise_std` | Noise standard deviation σ | 0.1 |
#
# ## Baseline Policies
#
# - **SAA**: selects the item with highest mean utility over available scenarios.
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{Features}]{x = [c_\text{base}; x_\text{raw}]}
# \fbox{Linear model}
# \xrightarrow{\theta \in \mathbb{R}^n}
# \fbox{argmax}
# \xrightarrow{y}
# ```
#
# **Model:** `Dense(n+d → n; bias=false)` — can in principle recover the exact mapping
# ``[I \mid W]`` from training data.
#
# **Maximizer:** `one_hot_argmax`.
