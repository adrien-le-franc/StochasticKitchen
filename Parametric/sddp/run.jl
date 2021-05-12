# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with SDDP

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

#set_processors(4)

include(joinpath(DIR, "optimize.jl"))

const oracle = SddpOracle()
const projection = HyperCubeProjection(peak_power)
step_size(k::Int64) = 1000/sqrt(k)
const parameters = SubgradientMethods.Parameters(zeros(48), 10, step_size, Dates.Minute(5), 0.01)

# run optimization

output = SubgradientMethods.optimize!(oracle, projection, parameters)

#set_processors(1)