# Understanding Benchmark Interface

This guide explains how benchmarks work through common interfaces in DecisionFocusedLearningBenchmarks.jl.
Understanding this interface is essential for using existing benchmarks and implementing new ones.

## Core Concepts

### DataSample Structure

All benchmarks work with [`DataSample`](@ref) objects that encapsulate the data needed for decision-focused learning:

```julia
@kwdef struct DataSample{I,F,S,C}
    x::F = nothing           # Input features  
    θ_true::C = nothing      # True cost/utility parameters
    y_true::S = nothing      # True optimal solution
    instance::I = nothing    # Problem instance object/additional data
end
```

The `DataSample` provides flexibility - not all fields need to be populated depending on the benchmark type and use case.

### Benchmark Type Hierarchy

The package defines a hierarchy of three abstract types:

```
AbstractBenchmark
└── AbstractStochasticBenchmark{exogenous}
    └── AbstractDynamicBenchmark{exogenous}
```

- **`AbstractBenchmark`**: static, single-stage optimization problems
- **`AbstractStochasticBenchmark{exogenous}`**: stochastic, single stage optimization problems
- **`AbstractDynamicBenchmark{exogenous}`**: multi-stage sequential decision-making problems

The `{exogenous}` type parameter indicates whether uncertainty distribution comes from external sources (`true`) or is influenced by decisions (`false`), which affects available methods.

## Common Interface Methods

### Data Generation

Every benchmark must implement a data generation method:

```julia
# Generate a single sample
generate_sample(benchmark::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample
```
This method should generate a single `DataSample` given a random number generator and optional parameters.

If needed, benchmarks can instead override the [`generate_dataset`](@ref) method to directly create the entire dataset:
```julia
generate_dataset(benchmark::AbstractBenchmark, size::Int; kwargs...) -> Vector{DataSample}
```

The default `generate_dataset` implementation calls `generate_sample` repeatedly, but benchmarks can override this for custom dataset generation logic.

### DFL Policy Components

Benchmarks provide the building blocks for decision-focused learning policies:

```julia
# Create a statistical model (e.g., a neural network)
generate_statistical_model(benchmark::AbstractBenchmark; kwargs...)

# Create an optimization maximizer/solver
generate_maximizer(benchmark::AbstractBenchmark; kwargs...)
```

The statistical model typically maps from features `x` to cost parameters `θ`.
The maximizer solves optimization problems given cost parameters `θ` (and potentially additional problem dependent keyword arguments), returning decision `y`.

### Benchmark Policies

Benchmarks can provide baseline policies for comparison and evaluation:

```julia
# Get baseline policies for comparison
generate_policies(benchmark::AbstractBenchmark) -> Tuple{Policy}
```
This returns a tuple of `Policy` objects representing different benchmark-specific policies:
```julia
struct Policy{F}
    name::String
    description::String  
    policy_function::F
end
```
A `Policy` is just a function with a name and description.

Policies can be evaluated across multiple instances/environments using:
```julia
evaluate_policy!(policy::Policy, instances; kwargs...) -> (rewards, data_samples)
```

### Evaluation Methods

Optional methods for analysis and visualization:

```julia
# Visualize data samples
plot_data(benchmark::AbstractBenchmark, sample::DataSample; kwargs...)
plot_instance(benchmark::AbstractBenchmark, instance; kwargs...)  
plot_solution(benchmark::AbstractBenchmark, sample::DataSample, solution; kwargs...)

# Compute optimality gap
compute_gap(benchmark::AbstractBenchmark, dataset, model, maximizer) -> Float64

# Evaluate objective value
objective_value(benchmark::AbstractBenchmark, sample::DataSample, solution)
```

## Benchmark-Specific Interfaces

### Static Benchmarks

Static benchmarks follow the basic interface above.

### Stochastic Benchmarks

Exogenous stochastic benchmarks add methods for scenario generation and anticipative solutions:

```julia
# Generate uncertainty scenarios (for exogenous benchmarks)
generate_scenario(benchmark::AbstractStochasticBenchmark{true}, instance; kwargs...)

# Solve anticipative optimization problem for given scenario
generate_anticipative_solution(benchmark::AbstractStochasticBenchmark{true}, 
                               instance, scenario; kwargs...)
```

### Dynamic Benchmarks

In order to model sequential decision-making, dynamic benchmarks additionally work with environments.
For this, they implement methods to create environments from instances or datasets:
```julia
# Create environment for sequential decision-making
generate_environment(benchmark::AbstractDynamicBenchmark, instance, rng; kwargs...) -> <:AbstractEnvironment

# Generate multiple environments
generate_environments(benchmark::AbstractDynamicBenchmark, dataset; kwargs...) -> Vector{<:AbstractEnvironment}
```
Similarly to `generate_dataset` and `generate_sample`, one only needs to implement `generate_environment`, as `generate_environments` has a default implementation that calls it repeatedly.

The [`AbstractEnvironment`](@ref) interface is defined as follows:
```julia
# Environment methods
get_seed(env::AbstractEnvironment)  # Get current RNG seed
reset!(env::AbstractEnvironment; reset_rng::Bool, seed=get_seed(env))  # Reset to initial state
observe(env::AbstractEnvironment) -> (obs, info)    # Get current observation  
step!(env::AbstractEnvironment, action) -> reward   # Take action, get reward
is_terminated(env::AbstractEnvironment) -> Bool     # Check if episode ended
```
