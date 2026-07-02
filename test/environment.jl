const DFLUtils = DecisionFocusedLearningBenchmarks.Utils

mutable struct MockEnv <: DFLUtils.AbstractEnvironment
    max_steps::Int
    step::Int
    state::Float64
end
MockEnv(; max_steps::Int=5) = MockEnv(max_steps, 0, 0.0)

DFLUtils.is_terminated(e::MockEnv) = e.step >= e.max_steps
DFLUtils.observe(e::MockEnv) = ([e.state], e.step)
function DFLUtils.reset!(e::MockEnv, rng::AbstractRNG)
    e.step = 0
    e.state = rand(rng)
    return nothing
end
function DFLUtils.step!(e::MockEnv, action, rng::AbstractRNG)
    e.step += 1
    r = rand(rng)
    e.state += r
    return r
end

function run_episode!(senv)
    reset_to_initial!(senv)
    rewards = Float64[]
    while !is_terminated(senv)
        push!(rewards, step!(senv, nothing))
    end
    return rewards
end

@testset "SeededEnvironment" begin
    @testset "Construction and accessors" begin
        e = MockEnv()
        senv = SeededEnvironment(e; seed=42)
        @test senv isa AbstractEnvironment
        @test senv.env === e
        @test get_seed(senv) == 42
        @test DFLUtils.get_rng(senv) isa Xoshiro
        @test occursin("seed=", sprint(show, senv))

        # Integer seeds are accepted and stored as UInt
        @test get_seed(SeededEnvironment(MockEnv(); seed=7)) === UInt(7)
        @test get_seed(SeededEnvironment(MockEnv(); seed=UInt(7))) === UInt(7)

        # Unseeded wrapper
        @test get_seed(SeededEnvironment(MockEnv())) === nothing

        # Explicit rng is used as-is
        myrng = Xoshiro(123)
        senv2 = SeededEnvironment(MockEnv(); seed=1, rng=myrng)
        @test DFLUtils.get_rng(senv2) === myrng
    end

    @testset "Delegation to wrapped env" begin
        senv = SeededEnvironment(MockEnv(; max_steps=3); seed=0)
        reset_to_initial!(senv)
        @test is_terminated(senv) == is_terminated(senv.env)
        @test observe(senv) == observe(senv.env)
        @test !is_terminated(senv)
        steps = 0
        while !is_terminated(senv)
            step!(senv, nothing)
            steps += 1
        end
        @test is_terminated(senv)
        @test steps == 3
    end

    @testset "reset! variants" begin
        # Integer seed: re-seeds the wrapper rng to that seed, then resets
        senv = SeededEnvironment(MockEnv(); seed=0)
        reset!(senv, 7)
        s7a = senv.env.state
        reset!(senv, 7)
        s7b = senv.env.state
        reset!(senv, 8)
        s8 = senv.env.state
        @test s7a == s7b
        @test s7a != s8
        @test s7a == rand(Xoshiro(7))   # confirms env.rng was seeded to 7 then drawn
        @test senv.env.step == 0

        # rng argument: resets using the provided rng, not the wrapper's
        e = MockEnv()
        senv = SeededEnvironment(e; seed=0)
        reset!(senv, Xoshiro(99))
        @test e.state == rand(Xoshiro(99))

        # no-arg reset!: uses the wrapper's current rng state (advances, does not re-seed)
        senv = SeededEnvironment(MockEnv(); seed=0)
        reset_to_initial!(senv)
        s1 = senv.env.state
        reset!(senv)
        s2 = senv.env.state
        @test s1 != s2

        # reset!(env, nothing) is equivalent to reset!(env) (no re-seeding)
        senv = SeededEnvironment(MockEnv(); seed=0)
        Random.seed!(DFLUtils.get_rng(senv), 123)
        reset!(senv, nothing)
        sa = senv.env.state
        Random.seed!(DFLUtils.get_rng(senv), 123)
        reset!(senv)
        sb = senv.env.state
        @test sa == sb
    end

    @testset "step! variants" begin
        # step! with explicit rng uses that rng
        senv = SeededEnvironment(MockEnv(); seed=0)
        reset!(senv, 0)
        r = step!(senv, nothing, Xoshiro(5))
        @test r == rand(Xoshiro(5))
        @test senv.env.step == 1

        # step! without rng uses the wrapper rng (reproducible from the same seed)
        senv = SeededEnvironment(MockEnv(); seed=0)
        reset_to_initial!(senv)
        r1 = step!(senv, nothing)
        reset_to_initial!(senv)
        r2 = step!(senv, nothing)
        @test r1 == r2
    end

    @testset "Reproducibility" begin
        # Same wrapper, repeated reset_to_initial!: identical episodes
        senv = SeededEnvironment(MockEnv(; max_steps=8); seed=123)
        ep1 = run_episode!(senv)
        ep2 = run_episode!(senv)
        @test ep1 == ep2
        @test length(ep1) == 8

        # Different seeds give different episodes
        senvA = SeededEnvironment(MockEnv(; max_steps=8); seed=1)
        senvB = SeededEnvironment(MockEnv(; max_steps=8); seed=2)
        @test run_episode!(senvA) != run_episode!(senvB)

        # reset_to_initial! on an unseeded wrapper does not error and resets the env
        senvU = SeededEnvironment(MockEnv())
        reset_to_initial!(senvU)
        @test senvU.env.step == 0
    end
end
