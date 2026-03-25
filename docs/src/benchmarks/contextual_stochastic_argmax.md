# Contextual Stochastic Argmax

[`ContextualStochasticArgmaxBenchmark`](@ref) is a minimalist contextual stochastic optimization benchmark problem.

The decision maker selects one item out of ``n``. Item values are uncertain at decision time: they depend on a base utility plus a context-correlated perturbation revealed only after the decision is made. An observable context vector, correlated with the perturbation via a fixed linear map ``W``, allows the learner to anticipate the perturbation and pick the right item.

## Problem Formulation

**Instance**: ``c_{\text{base}} \sim \mathcal{U}[0,1]^n``, base values for ``n`` items.

**Context**: ``x_{\text{raw}} \sim \mathcal{N}(0, I_d)``, a ``d``-dimensional signal correlated with item values. The feature vector passed to the model is ``x = [c_{\text{base}};\, x_{\text{raw}}] \in \mathbb{R}^{n+d}``.

**Scenario**: the realized item values are
```math
\xi = c_{\text{base}} + W x_{\text{raw}} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)
```
where ``W \in \mathbb{R}^{n \times d}`` is a fixed matrix unknown to the learner.

**Decision**: ``y \in \{e_1, \ldots, e_n\}`` (one-hot vector selecting one item).

## Policies

### DFL Policy

```math
\xrightarrow[\text{Features}]{x}
\fbox{Neural network $\varphi_w$}
\xrightarrow[\text{Predicted values}]{\hat{\theta}}
\fbox{\texttt{one\_hot\_argmax}}
\xrightarrow[\text{Decision}]{y}
```

The neural network predicts item values ``\hat{\theta} \in \mathbb{R}^n`` from the feature vector ``x \in \mathbb{R}^{n+d}``. The default architecture is `Dense(n+d => n; bias=false)`, which can exactly recover the optimal linear predictor ``[I_n \mid W]``, so a well-trained model should reach near-zero gap.

### SAA Policy

``y_{\text{SAA}} = \operatorname{argmax}\bigl(\frac{1}{S}\sum_s \xi^{(s)}\bigr)`` — the exact SAA-optimal decision for linear argmax, accessible via `generate_baseline_policies(bench).saa`.
