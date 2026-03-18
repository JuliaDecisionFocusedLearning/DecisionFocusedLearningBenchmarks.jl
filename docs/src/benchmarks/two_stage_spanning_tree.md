# Problem statement

We consider a two-stage stochastic variant of the classic [minimum spanning tree problem](https://en.wikipedia.org/wiki/Minimum_spanning_tree).

Rather than immediately constructing a spanning tree and incurring a cost ``c_e`` for each selected edge in the tree, we instead can build only a partial tree (forest) during the first stage and paying first stage costs ``c_e`` for the selected edges. Then, second stage costs ``d_e`` are revealed and replace first stage costs. The task then involves completing the first stage forest into a spanning tree.

The objective is to minimize the total incurred cost in expectation.

## Instance
Let ``G = (V,E)`` be an undirected **graph**, and ``S`` be a finite set of **scenarios**.

For each edge ``e`` in ``E``, we have a **first stage cost** ``c_e\in\mathbb{R}``.

For each edge ``e`` in ``E`` and scenario ``s`` in ``S``, we have a **second stage cost** ``d_{es}\in\mathbb{R}``.

# MIP formulation
Unlike the regular minimum spanning tree problem, this two-stage variant is NP-hard.
However, it can still be formulated as linear program with binary variables, and exponential number of constraints.
```math
\begin{array}{lll}
\min\limits_{y, z}\, & \sum\limits_{e\in E}c_e y_e + \frac{1}{|S|}\sum\limits_{s \in S}d_{es}z_{es} & \\
\mathrm{s.t.}\, & \sum\limits_{e\in E}y_e + z_{es} = |V| - 1, & \forall s \in S\\
& \sum\limits_{e\in E(Y)} y_e + z_{es} \leq |Y| - 1,\quad & \forall \emptyset \subsetneq Y \subsetneq V,\, \forall s\in S\\
& y_e\in \{0, 1\}, & \forall e\in E\\
& z_{es}\in \{0, 1\}, & \forall e\in E, \forall s\in S
\end{array}
```
where ``y_e`` is a binary variable indicating if ``e`` is in the first stage solution, and ``z_{es}`` is a binary variable indicating if ``e`` is in the second stage solution for scenario ``s``.
