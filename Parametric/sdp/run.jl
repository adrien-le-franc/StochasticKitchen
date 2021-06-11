# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with SDDP

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include(joinpath(DIR, "optimize.jl"))

const oracle = ParametricMultistageOracle(model)
const projection = HyperCubeProjection(peak_power)
step_size(k::Int64) = 500/sqrt(k)
const parameters = SubgradientMethods.Parameters(zeros(48), 100, step_size, Dates.Minute(5), 0.01)

# run optimization

output = SubgradientMethods.optimize!(oracle, projection, parameters)
#save("/home/SubgradientMethods/examples/toy/output.jld2", "output", output)

set_processors(1)