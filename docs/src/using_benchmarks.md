# Using Benchmarks

This guide covers everything you need to work with existing benchmarks in DecisionFocusedLearningBenchmarks.jl: generating datasets, assembling DFL pipeline components, applying algorithms, and evaluating results.

---

## What is a benchmark?

A benchmark bundles a problem family (an instance generator, a combinatorial solver, and a statistical model architecture) into a single object. It provides everything needed to run a Decision-Focused Learning experiment out of the box, without having to create each component from scratch.
Three abstract types cover the main settings:
- **`AbstractStaticBenchmark`**: static problems (one instance, one decision)
- **`AbstractStochasticBenchmark{exogenous}`**: stochastic problems (type parameter indicates whether uncertainty is exogenous)
- **`AbstractDynamicBenchmark{exogenous}`**: sequential / multi-stage problems

The sections below explain what changes between these settings. For most purposes, start with a static benchmark to understand the core workflow.

---

## Core workflow

Every benchmark exposes three key methods. For any static benchmark:

```julia
bench = ArgmaxBenchmark()
model = generate_statistical_model(bench; seed=0)   # Flux model
maximizer = generate_maximizer(bench)               # combinatorial oracle
dataset = generate_dataset(bench, 100; seed=0)      # Vector{DataSample}
```

- **`generate_statistical_model`**: returns an untrained neural network that maps input features `x` to cost parameters `θ`.
- **`generate_maximizer`**: returns a callable `(θ; context...) -> y` that solves the combinatorial problem given cost parameters.
- **`generate_dataset`**: returns labeled training data as a `Vector{DataSample}`.

At inference time these two pieces compose naturally as an end-to-end policy:

```julia
θ = model(sample.x)                  # predict cost parameters
y = maximizer(θ; sample.context...)  # solve the optimization problem
```

---

## DataSample at a glance

All data in the package is represented as [`DataSample`](@ref) objects.

| Field | Type | Description |
|-------|------|-------------|
| `x` | any | Input features (fed to the statistical model) |
| `θ` | any | Intermediate cost parameters |
| `y` | any | Output decision / solution |
| `context` | `NamedTuple` | Solver kwargs spread into `maximizer(θ; sample.context...)` |
| `extra` | `NamedTuple` | Non-solver data (scenario, reward, step, …), never passed to the solver |

Not all fields are populated in every sample, depending on the setting. For convenience, named entries inside `context` and `extra` can be accessed directly on the sample via property forwarding:

```julia
sample.instance   # looks up :instance in context first, then in extra
sample.scenario   # looks up :scenario in context first, then in extra
```

---

## Benchmark type specifics

### Static benchmarks

For static benchmarks (`<:AbstractStaticBenchmark`), `generate_dataset` may compute a default ground-truth label `y` if the benchmark implements it:

```julia
bench = ArgmaxBenchmark()
dataset = generate_dataset(bench, 100; seed=0)   # Vector{DataSample} with x, y, context
```

You can override the labels by providing a `target_policy`:

```julia
my_policy = sample -> DataSample(; sample.context..., x=sample.x, y=my_algorithm(sample.instance))
dataset = generate_dataset(bench, 100; seed=0, target_policy=my_policy)
```

### Stochastic benchmarks (exogenous)

For `AbstractStochasticBenchmark{true}` benchmarks the default call returns *unlabeled* samples, each sample carries one scenario in `sample.extra.scenario`:

```julia
bench   = StochasticVehicleSchedulingBenchmark()
dataset = generate_dataset(bench, 20; seed=0)   # y = nothing
```

Request multiple scenarios per instance with `nb_scenarios`:

```julia
dataset = generate_dataset(bench, 20; seed=0, nb_scenarios=5)
# returns 20 × 5 = 100 samples
```

To compute labels, wrap your algorithm as a `target_policy`:

```julia
anticipative = generate_anticipative_solver(bench)   # (scenario; kwargs...) -> y

policy = (sample, scenarios) -> [
    DataSample(; sample.context..., x=sample.x,
                 y=anticipative(ξ; sample.context...))
    for ξ in scenarios
]
labeled = generate_dataset(bench, 20; seed=0, nb_scenarios=5, target_policy=policy)
```

### Dynamic benchmarks

Dynamic benchmarks use a two-step workflow:

```julia
bench = DynamicVehicleSchedulingBenchmark()

# Step 1: create environments (reusable across experiments)
envs = generate_environments(bench, 10; seed=0)

# Step 2: roll out a policy to collect training trajectories
policy = generate_baseline_policies(bench)[1]          # e.g. lazy policy
dataset = generate_dataset(bench, envs; target_policy=policy)
# dataset is a flat Vector{DataSample} of all steps across all trajectories
```

`target_policy` is **required** to create datasets for dynamic benchmarks (there is no default label).
It must be a callable `(env) -> Vector{DataSample}` that performs a full episode
rollout and returns the resulting trajectory.

---

## Seed / RNG control

All `generate_dataset` and `generate_environments` calls accept either `seed`
(creates an internal `MersenneTwister`) or `rng` for full control:

```julia
using Random
rng = MersenneTwister(42)
dataset = generate_dataset(bench, 50; rng=rng)
```

---

## Evaluation

```julia
# Average relative optimality gap across a dataset
gap = compute_gap(bench, dataset, model, maximizer)
```

# Objective value for a single decision
```julia
obj = objective_value(bench, sample, y)
```

---

## Baseline policies

`generate_baseline_policies` returns a collection of named callables that can serve as
reference points or as `target_policy` arguments:

```julia
policies = generate_baseline_policies(bench)
pol = policies[1]   # e.g. greedy, lazy, or anticipative policy
```

- **Static / stochastic:** `pol(sample) -> DataSample`
- **Dynamic:** `pol(env) -> Vector{DataSample}` (full episode trajectory)

For dynamic benchmarks you can evaluate a policy over multiple episodes:

```julia
rewards, samples = evaluate_policy!(pol, envs, n_episodes)
```

---

## Visualization

Plots is an **optional** dependency, load it with `using Plots` to unlock the plot functions. Not all benchmarks support visualization, call `has_visualization(bench)` to check.
```julia
using Plots

bench = Argmax2DBenchmark()
dataset = generate_dataset(bench, 10)
sample = dataset[1]

has_visualization(bench)           # true
plot_instance(bench, sample)       # problem geometry only
plot_solution(bench, sample)       # sample.y overlaid on the instance
plot_solution(bench, sample, y)    # convenience 3-arg form: override y before plotting

# Dynamic benchmarks only
traj = generate_anticipative_solver(bench)(env)
plot_trajectory(bench, traj)           # grid of epoch subplots
anim = animate_trajectory(bench, traj; fps=2)
gif(anim, "episode.gif")
```

- `has_visualization(bench)`: returns `true` for benchmarks that implement plot support (if Plots is loaded).
- `plot_instance(bench, sample; kwargs...)`: renders the problem geometry without any solution.
- `plot_solution(bench, sample; kwargs...)`: renders `sample.y` overlaid on the instance.
- `plot_solution(bench, sample, y; kwargs...)`: 3-arg convenience form that overrides `y` before plotting.
- `plot_trajectory(bench, traj; kwargs...)`: dynamic benchmarks only; produces a grid of per-epoch subplots.
- `animate_trajectory(bench, traj; kwargs...)`: dynamic benchmarks only, returns a `Plots.Animation` that can be saved with `gif(anim, "file.gif")`.
