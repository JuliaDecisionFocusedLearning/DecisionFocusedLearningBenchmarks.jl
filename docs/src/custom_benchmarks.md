# Creating Custom Benchmarks

This guide explains how to implement new benchmarks in
DecisionFocusedLearningBenchmarks.jl. It is aimed at developers who want to add
problems to the benchmark suite or integrate their own domains.

---

## Type hierarchy

```
AbstractBenchmark
├── AbstractStaticBenchmark
├── AbstractStochasticBenchmark{exogenous}
└── AbstractDynamicBenchmark{exogenous}
```

| Type | Use case |
|------|----------|
| `AbstractStaticBenchmark` | Static, single-stage optimization (e.g. shortest path, portfolio) |
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

## `AbstractStaticBenchmark`: required methods

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
generate_baseline_policies(bench::MyBenchmark) -> collection of callables
is_minimization_problem(bench::MyBenchmark) -> Bool   # default: true (minimization)
objective_value(bench::MyBenchmark, sample::DataSample, y) -> Real
compute_gap(bench::MyBenchmark, dataset, model, maximizer) -> Float64
has_visualization(bench::MyBenchmark) -> Bool                            # default: false; return true when plot methods are implemented/available
plot_context(bench::MyBenchmark, sample::DataSample; kwargs...)
plot_sample(bench::MyBenchmark, sample::DataSample; kwargs...)
```

---

## `AbstractStochasticBenchmark{true}`

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

## `AbstractDynamicBenchmark`

Dynamic benchmarks extend stochastic ones with an environment-based rollout interface.

### Environment generation

```julia
# Option A: build one bare environment (the framework wraps it in a SeededEnvironment)
build_environment(bench::MyDynamicBenchmark, rng::AbstractRNG; kwargs...) -> AbstractEnvironment

# Option B: override when environments are not independent (e.g. loaded from files)
generate_environments(bench::MyDynamicBenchmark, n::Int; seed=nothing, kwargs...) -> Vector{SeededEnvironment}
```

You implement **`build_environment`**, which returns a *bare* environment. The package then automatically wraps it in a [`SeededEnvironment`](@ref) (attaching a seed and RNG) through two public entry points:

```julia
generate_environment(bench; seed=nothing, kwargs...)      -> SeededEnvironment       # one env
generate_environments(bench, n; seed=nothing, kwargs...)  -> Vector{SeededEnvironment} # n envs
```

For environments that cannot be drawn independently (e.g. loaded from files), override `generate_environments` instead of implementing `build_environment`.
An override must return already-wrapped `SeededEnvironment`s (use `SeededEnvironment(env; seed=...)`).
Do not override `generate_environment`: it just delegates to `generate_environments`.

### `AbstractEnvironment` interface

Your environment must be **stateless about randomness**: it does *not* store its own RNG or seed.
All randomness is owned by the wrapping [`SeededEnvironment`](@ref), which passes its `rng` into every `reset!` and `step!`.
Draw all stochasticity from that `rng` so that re-seeding the wrapper (via [`reset_to_initial!`](@ref)) replays an episode exactly.

```julia
reset!(env::MyEnv, rng::AbstractRNG)                    # Reset to a starting state, sampling from rng
observe(env::MyEnv) -> (observation, state)             # Current observation and internal state
step!(env::MyEnv, action, rng::AbstractRNG) -> reward   # Apply action, advance state (draw from rng)
is_terminated(env::MyEnv) -> Bool                       # True when the episode has ended
```

Environments may ignore the `rng` argument.

### Baseline policies (required for `generate_dataset`)

```julia
generate_baseline_policies(bench::MyDynamicBenchmark)
# Returns named callables: (env) -> Vector{DataSample}
# Each callable performs a full episode rollout and returns the trajectory.
```

### Anticipative solver (optional)

```julia
generate_anticipative_solver(bench::MyDynamicBenchmark)
# Returns a callable: (env; reset_env=true, kwargs...) -> Vector{DataSample}
# reset_env=true  → reset environment before solving
# reset_env=false → solve from current state
```

### Optional visualization methods

```julia
plot_trajectory(bench::MyDynamicBenchmark, traj::Vector{DataSample}; kwargs...)
animate_trajectory(bench::MyDynamicBenchmark, traj::Vector{DataSample}; kwargs...)
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
