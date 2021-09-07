# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with SDDP

DIR = @__DIR__

using Statistics
include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include(joinpath(DIR, "optimize.jl"))


const oracle = SddpOracle(80)
const projection = HyperCubeProjection(peak_power)
step_size(k::Int64) = 1000/k
const parameters = SubgradientMethods.Parameters(zeros(48), step_size, 100, Dates.Minute(2), 0.01)

# run optimization

output = SubgradientMethods.optimize!(oracle, projection, parameters)

# log
log = Dict(
    "x_opt" => output.variable,
    "f_values" => output.all_values,
    "n_steps" => output.final_iteration,
    "overall_time" => round(output.elapsed, Dates.Second),
    "average_time_per_gradient_call" => mean(output.elapsed_per_oracle_call[2:end]),
    "params" => params)

save("/home/StochasticKitchen/Parametric/results/sddp/log_tx0.jld2", "log", log)


set_processors(1)