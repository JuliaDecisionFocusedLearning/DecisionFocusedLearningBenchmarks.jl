# @testitem "DVSP - parsing" begin
#     using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling:
#         read_vsp_instance, location_count, customer_count
#     path = joinpath(@__DIR__, "data", "vsp_instance.txt")
#     instance = read_vsp_instance(path)
#     @test location_count(instance) == 6
#     @test customer_count(instance) == 5
# end
