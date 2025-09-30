# Portfolio Optimization

[`PortfolioOptimizationBenchmark`](@ref) is a Markovitz portfolio optimization problem, where asset prices are unknown, and only contextual data is available to predict these prices.
The goal is to predict asset prices $c$ and maximize the expected return of a portfolio, subject to a risk constraint using this maximization program:
```math
\begin{aligned}
\max\quad & c^\top x\\
\text{s.t.}\quad & x^\top \Sigma x \leq \gamma\\
& 1^\top x \leq 1\\
& x \geq 0
\end{aligned}
```

!!! warning
    Documentation for this benchmark is still under development. Please refer to the source code and API for more details.