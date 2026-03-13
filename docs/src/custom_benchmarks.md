# Creating Custom Benchmarks

This guide explains how to implement new benchmarks in
DecisionFocusedLearningBenchmarks.jl. It is aimed at developers who want to add
problems to the benchmark suite or integrate their own domains.

---

## Type hierarchy

```
AbstractBenchmark
└── AbstractStochasticBenchmark{exogenous}
    └── AbstractDynamicBenchmark{exogenous}
```

| Type | Use case |
|------|----------|
| `AbstractBenchmark` | Static, single-stage optimization (e.g. shortest path, portfolio) |
| `AbstractStochasticBenchmark{true}` | Single-stage with exogenous uncertainty (scenarios drawn independently of decisions) |
| `AbstractStochasticBenchmark{false}` | Single-stage with endogenous uncertainty |
| `AbstractDynamicBenchmark{true}` | Multi-stage sequential decisions with exogenous uncertainty |
| `AbstractDynamicBenchmark{false}` | Multi-stage sequential decisions with endogenous uncertainty |

---

## Implementation strategies

There are three strategies for data generation. Pick the one that best fits your
benchmark:

| Strategy | Method to implement | When to use |
|----------|---------------------|-------------|
| **1** | `generate_instance(bench, rng; kwargs...) -> DataSample` | Samples are independent; `y=nothing` at generation time |
| **2** | `generate_sample(bench, rng; kwargs...) -> DataSample` | Samples are independent; you want to compute `y` inside `generate_sample` |
| **3** | `generate_dataset(bench, N; kwargs...) -> Vector{DataSample}` | Samples are not independent (e.g. loaded from shared files) |

The default `generate_sample` calls `generate_instance` and then applies
`target_policy` to the returned sample. `generate_dataset` calls `generate_sample`
repeatedly and applies `target_policy` to each result.

---

## `AbstractBenchmark`: required methods

### Data generation (choose one strategy)

```julia
# Strategy 1: recommended for most static benchmarks
generate_instance(bench::MyBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

# Strategy 2: when you want to compute y inside generate_sample
generate_sample(bench::MyBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

# Strategy 3: when samples are not independent
generate_dataset(bench::MyBenchmark, N::Int; kwargs...) -> Vector{DataSample}
```

### Pipeline components (required)

```julia
generate_statistical_model(bench::MyBenchmark; seed=nothing)
# Returns an untrained Flux model mapping x -> θ

generate_maximizer(bench::MyBenchmark)
# Returns a callable (θ; context...) -> y
```

### Optional methods

```julia
is_minimization_problem(bench::MyBenchmark) -> Bool   # default: false (maximization)
objective_value(bench::MyBenchmark, sample::DataSample, y) -> Real
compute_gap(bench::MyBenchmark, dataset, model, maximizer) -> Float64
plot_data(bench::MyBenchmark, sample::DataSample; kwargs...)
plot_instance(bench::MyBenchmark, instance; kwargs...)
plot_solution(bench::MyBenchmark, sample::DataSample, y; kwargs...)
generate_baseline_policies(bench::MyBenchmark) -> collection of callables
```

---

## `AbstractStochasticBenchmark{true}`: additional methods

For stochastic benchmarks with exogenous uncertainty, implement:

```julia
# Instance + features, no scenario (y = nothing)
generate_instance(bench::MyStochasticBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

# Draw one scenario given the instance encoded in context
generate_scenario(bench::MyStochasticBenchmark, rng::AbstractRNG; context...) -> scenario
# Note: sample.context is spread as kwargs when called
```

#### Anticipative solver (optional)

```julia
generate_anticipative_solver(bench::MyStochasticBenchmark)
# Returns a callable: (scenario; context...) -> y
```

#### `DataSample` conventions

- `context`: solver kwargs (instance data, graph, capacities, …)
- `extra`: scenario: **never** passed to the maximizer

```julia
DataSample(; x=features, y=nothing,
             instance=my_instance,      # goes into context
             extra=(; scenario=ξ))
```

---

## `AbstractDynamicBenchmark`: additional methods

Dynamic benchmarks extend stochastic ones with an environment-based rollout interface.

### Environment generation

```julia
# Strategy A: generate one environment at a time (default implementation of
#              generate_environments calls this repeatedly)
generate_environment(bench::MyDynamicBenchmark, rng::AbstractRNG; kwargs...) -> AbstractEnvironment

# Strategy B: override when environments are not independent (e.g. loaded from files)
generate_environments(bench::MyDynamicBenchmark, n::Int; rng, kwargs...) -> Vector{<:AbstractEnvironment}
```

### `AbstractEnvironment` interface

Your environment type must implement:

```julia
get_seed(env::MyEnv)                             # Return the RNG seed used at creation
reset!(env::MyEnv; reset_rng::Bool, seed=get_seed(env))  # Reset to initial state
observe(env::MyEnv) -> (observation, info)       # Current observation
step!(env::MyEnv, action) -> reward              # Apply action, advance state
is_terminated(env::MyEnv) -> Bool                # True when episode has ended
```

### Baseline policies (required for `generate_dataset`)

```julia
generate_baseline_policies(bench::MyDynamicBenchmark)
# Returns named callables: (env) -> Vector{DataSample}
# Each callable performs a full episode rollout and returns the trajectory.
```

`generate_dataset` for dynamic benchmarks **requires** a `target_policy` kwarg, 
there is no default. The `target_policy` must be a callable `(env) -> Vector{DataSample}`.

### `DataSample` conventions

- `context`: solver-relevant state (observation fields, graph, etc.)
- `extra`: reward, step counter, history (never passed to the maximizer)

```julia
DataSample(; x=features, y=action,
             instance=current_state,             # goes into context
             extra=(; reward=r, step=t))
```

---

## `DataSample` construction guide

| Benchmark category | `context` fields | `extra` fields |
|--------------------|-----------------|----------------|
| Static | instance, graph, capacities, … | — |
| Stochastic | instance (not scenario) | `scenario` |
| Dynamic | solver-relevant state / observation | `reward`, `step`, `history`, … |

Any named argument that is not `x`, `θ`, `y`, `context`, or `extra` is treated as a `context` field:

```julia
# Equivalent forms:
DataSample(; x=feat, y=sol, instance=inst)
DataSample(; x=feat, y=sol, context=(; instance=inst))

# With extra:
DataSample(; x=feat, y=nothing, instance=inst, extra=(; scenario=ξ))
```

Keys must not appear in both `context` and `extra`, the constructor raises an error.
