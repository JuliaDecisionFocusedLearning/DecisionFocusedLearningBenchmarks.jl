function build_polytope(N; shift=0.0)
    return [[cospi(2k / N + shift), sinpi(2k / N + shift)] for k in 0:(N - 1)]
end
