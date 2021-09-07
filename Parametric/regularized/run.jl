# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with SDDP

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include(joinpath(DIR, "optimize.jl"))

const oracle = ParametricMultistageOracle(model)
const projection = HyperCubeProjection(peak_power)
step_size(k::Int64) = 1000/k
const parameters = SubgradientMethods.Parameters(zeros(48), 100, step_size, Dates.Minute(30), 0.01)

# run optimization

output = SubgradientMethods.optimize!(oracle, projection, parameters)

set_processors(1)

# log
log = Dict(
    "x_opt" => output.variable,
    "f_values" => output.all_values,
    "n_steps" => output.final_iteration,
    "overall_time" => output.elapsed,
    "average_time_per_gradient_call" => nothing)

save("/home/StochasticKitchen/Parametric/results/regularized/log.jld2", "log", log)