# # Contextual Stochastic Argmax
# Select the best item when utilities are random but correlated with observable context:
# a linear model must learn the mapping from context to expected utilities.

using DecisionFocusedLearningBenchmarks
using Plots

b = ContextualStochasticArgmaxBenchmark()

# Stochastic benchmarks need a labeling policy to generate training targets.
# We use the anticipative oracle: given realized scenario ξ it returns the best item.
anticipative = generate_anticipative_solver(b)
policy = (ctx, scenarios) -> [
    DataSample(; ctx.context..., x=ctx.x, y=anticipative(ξ), extra=(; scenario=ξ))
    for ξ in scenarios
]
dataset = generate_dataset(b, 20; target_policy=policy, seed=0)
sample = first(dataset)

# ## Observable input
#
# At inference time `c_base` and `x_raw` are known (not the realized utility vector ξ).
# `plot_instance` shows the base utilities `c_base`:
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
# Left: realized scenario ξ. Right: selected item (red):
plot_solution(b, sample)

# ## Untrained policy

# A DFL policy chains two components: a statistical model predicting expected item utilities:
model = generate_statistical_model(b)     # linear map: features → predicted expected utilities
# and a maximizer selecting the item with the highest predicted utility:
maximizer = generate_maximizer(b)         # one-hot argmax

# A randomly initialized policy selects items with no relation to their expected utilities.
# Left: predicted utilities θ̂. Right: selected item (red):
θ_pred = model(sample.x)
plot_solution(b, DataSample(; sample.context..., x=sample.x, θ=θ_pred, y=maximizer(θ_pred)))

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
# A linear model ``\hat{\theta} = [I \mid W] \cdot x`` can exactly recover the optimal
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
# \xrightarrow{\hat{\theta} \in \mathbb{R}^n}
# \fbox{argmax}
# \xrightarrow{y}
# ```
#
# **Model:** `Dense(n+d → n; bias=false)` — can in principle recover the exact mapping
# ``[I \mid W]`` from training data.
#
# **Maximizer:** `one_hot_argmax`.
