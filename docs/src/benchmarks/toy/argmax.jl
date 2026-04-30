# # Argmax
# Select the single best item from a set of `n` items, given features correlated with hidden
# item scores. This is a minimalist DFL setting: equivalent to multiclass
# classification, but with an argmax layer instead of softmax. Useful as a minimal sandbox for
# understanding DFL concepts.

using DecisionFocusedLearningBenchmarks
using Plots
using Statistics

b = ArgmaxBenchmark(; seed=0)

# ## Observable input
#
# At inference time the decision-maker observes only a feature matrix `x`
# (rows = features, columns = items):
dataset = generate_dataset(b, 100; seed=0)
sample = first(dataset)
plot_instance(b, sample)

# ## A training sample
#
# Each sample is a labeled triple `(x, θ, y)`:
# - `x`: feature matrix (observable at train and test time)
# - `θ`: true item scores (training supervision only, hidden at test time)
# - `y`: optimal one-hot decision derived from `θ`
#
# The full training triple (features, true scores, and optimal decision):
plot_sample(b, sample)

# ## Untrained policy

# A DFL policy chains two components: a statistical model predicting scores from features:
model = generate_statistical_model(b)     # linear map: features → predicted scores
# and a maximizer turning those scores into a decision:
maximizer = generate_maximizer(b)         # one-hot argmax

# A randomly initialized policy makes essentially random decisions:
θ_pred = model(sample.x)
y_pred = maximizer(θ_pred)
#
plot_sample(b, DataSample(sample; θ=θ_pred, y=y_pred))

# The goal of training is to find parameters that maximize accuracy.
# Current accuracy on the dataset:
mean(maximizer(model(s.x)) == s.y for s in dataset)

# ---
# ## Problem Description
#
# In the **Argmax benchmark**, a feature matrix ``x \in \mathbb{R}^{p \times n}`` is
# observed. A hidden linear encoder maps ``x`` to a score vector
# ``\theta = \text{encoder}(x) \in \mathbb{R}^n``. The task is to select the item with
# the highest score:
# ```math
# y = \mathrm{argmax}(\theta)
# ```
# The solution ``y`` is encoded as a one-hot vector.
# The score vector ``\theta`` is never observed (only features ``x`` are available).
# The DFL pipeline trains a model ``f_w`` so that ``\mathrm{argmax}(f_w(x))`` matches
# ``\mathrm{argmax}(\theta)`` at decision time.
#
# ## Key Parameters
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `instance_dim` | Number of items | 10 |
# | `nb_features` | Feature dimension `p` | 5 |
#
# ## DFL Policy
#
# ```math
# \xrightarrow[\text{Features}]{x \in \mathbb{R}^{p \times n}}
# \fbox{Linear model $f_w$}
# \xrightarrow[\text{Predicted scores}]{\theta \in \mathbb{R}^n}
# \fbox{argmax}
# \xrightarrow[\text{Selection}]{y \in \{0,1\}^n}
# ```
#
# **Model:** `Chain(Dense(nb_features → 1; bias=false), vec)`: a single linear layer
# predicting one score per item.
#
# **Maximizer:** `one_hot_argmax`: returns a one-hot vector at the argmax index.
