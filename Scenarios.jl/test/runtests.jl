using Scenarios
using CSV
using Test

df = CSV.read(joinpath(@__DIR__, "data/1.csv"))

support = Scenarios.compute_support_of_error_process(df, [1, 4, 8, 16, 48, 96], 5) 
