"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
struct TwoStageSpanningTreeInstance{T}
    "Graph"
    graph::SimpleGraph{Int}
    "First stage costs for each edge"
    first_stage_costs::Vector{T}
    "Second stage costs for each edge and scenario [e, s]"
	second_stage_costs::Matrix{T}
end

Base.show(io::IO, instance::TwoStageSpanningTreeInstance) = print(io, "$(collect(edges(instance.graph))), $(instance.first_stage_costs), $(instance.second_stage_costs)")

function TwoStageSpanningTreeInstance(; n, m, nb_scenarios=1, c_range=1:20, d_range=1:20, seed=nothing, type=Float64)
	g = Graphs.grid((n, m))
	rng = MersenneTwister(seed)
	c = type[rand(rng, c_range) for _ in 1:ne(g)]
	d = type[rand(rng, d_range) for _ in 1:ne(g), _ in 1:nb_scenarios]

	return TwoStageSpanningTreeInstance(g, c, d)
end

"""
$TYPEDSIGNATURES

Return the number of scenarios of `instance`.
"""
nb_scenarios(instance::TwoStageSpanningTreeInstance) = size(instance.second_stage_costs, 2)
