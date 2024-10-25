# Subset Selection

[`SubsetSelectionBenchmark`](@ref) is the most trivial benchmark problem in this package.
It is minimalistic and serves as a simple example for debugging and testing purposes.

## Description
We have a set of ``n`` items, each item having an unknown value.
We want to select a subset of ``k`` items that maximizes the sum of the values of the selected items.

As input, instead of the items costs, we are given a feature vector, such that an unknown linear mapping between the feature vector and the value of the items exists.

By default, this linear mapping is the identity mapping, i.e., the value of each item is equal to the value of the corresponding feature vector element.
However, this mapping can be changed by setting the `identity_mapping` parameter to false.
