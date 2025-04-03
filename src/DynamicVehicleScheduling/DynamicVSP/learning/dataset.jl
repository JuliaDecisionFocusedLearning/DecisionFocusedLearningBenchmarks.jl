function load_VSP_dataset(
    datadir::String; model_builder=highs_model, use_2D_features=false, kwargs...
)
    instances_files = filtered_readdir(datadir)
    X = Tuple{Matrix{Float32},VSPState{VSPInstance{Float64}}}[]
    Y = BitMatrix[]

    for (i, f) in enumerate(instances_files)
        static_instance = read_vsp_instance((joinpath(datadir, f)))
        env = DVSPEnv(static_instance; seed=i, kwargs...)

        # Compute the anticipative policy
        routes_anticipative = anticipative_solver(env; model_builder)
        reset!(env)
        for e in eachindex(routes_anticipative)
            next_epoch!(env)
            # Store the state
            state = env.state
            features = Matrix(
                use_2D_features ? compute_2D_features(env) : compute_features(env)
            )
            push!(X, (features, state))
            routes = routes_anticipative[e]
            # Store the solution
            push!(
                Y,
                VSPSolution(
                    state_route_from_env_routes(env, routes);
                    max_index=nb_locations(state.instance),
                ).edge_matrix,
            )
            # Update the environment
            apply_decision!(env, routes)
        end
    end
    return X, Y
end
