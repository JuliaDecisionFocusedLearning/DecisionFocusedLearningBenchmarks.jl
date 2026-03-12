# Using Benchmarks

This guide covers everything you need to work with existing benchmarks in
DecisionFocusedLearningBenchmarks.jl: generating datasets, assembling DFL pipeline
components, and evaluating results.

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

Not all fields are populated in every sample. For convenience, named entries inside
`context` and `extra` can be accessed directly on the sample via property forwarding:

```julia
sample.instance   # looks up :instance in context first, then in extra
sample.scenario   # looks up :scenario in context first, then in extra
```

---

## Generating datasets for training

### Static benchmarks

For static benchmarks (`<:AbstractBenchmark`) the framework already computes the
ground-truth label `y`:

```julia
bench = ArgmaxBenchmark()
dataset = generate_dataset(bench, 100; seed=0)   # Vector{DataSample} with x, y, context
```

You can override the labels by providing a `target_policy`:

```julia
my_policy = sample -> DataSample(; sample.context..., x=sample.x,
                                   y=my_algorithm(sample.instance))
dataset = generate_dataset(bench, 100; seed=0, target_policy=my_policy)
```

### Stochastic benchmarks (exogenous)

For `AbstractStochasticBenchmark{true}` benchmarks the default call returns
*unlabeled* samples, each sample carries one scenario in `sample.extra.scenario`:

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

# Step 1 — create environments (reusable across experiments)
envs = generate_environments(bench, 10; seed=0)

# Step 2 — roll out a policy to collect training trajectories
policy = generate_baseline_policies(bench)[1]          # e.g. lazy policy
dataset = generate_dataset(bench, envs; target_policy=policy)
# dataset is a flat Vector{DataSample} of all steps across all trajectories
```

`target_policy` is **required** for dynamic benchmarks (there is no default label).
It must be a callable `(env) -> Vector{DataSample}` that performs a full episode
rollout and returns the resulting trajectory.

### Seed / RNG control

All `generate_dataset` and `generate_environments` calls accept either `seed`
(creates an internal `MersenneTwister`) or `rng` for full control:

```julia
using Random
rng = MersenneTwister(42)
dataset = generate_dataset(bench, 50; rng=rng)
```

---

## DFL pipeline components

```julia
model = generate_statistical_model(bench; seed=0)   # untrained Flux model
maximizer = generate_maximizer(bench)                   # combinatorial oracle
```

These two pieces compose naturally:

```julia
θ = model(sample.x)                  # predict cost parameters
y = maximizer(θ; sample.context...)      # solve the optimization problem
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

Where implemented, benchmarks provide benchmark-specific plotting helpers:

```julia
plot_data(bench, sample)            # overview of a data sample
plot_instance(bench, instance)      # raw problem instance
plot_solution(bench, sample, y)     # overlay solution on instance
```
