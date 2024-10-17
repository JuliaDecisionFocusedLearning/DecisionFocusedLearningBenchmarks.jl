# Subset Selection

[`SubsetSelectionBenchmark`](@ref) is a very simple benchmark problem of subset selection.

We have a set of ``n`` items, each item having an `unknown' value.
We want to select a subset of ``k`` items that maximizes the sum of the values of the selected items.

As input, we are given a feature vector, that contains exactly the value of each item.
The goal is to learn the identity mapping between the feature vector and the value of the items.
