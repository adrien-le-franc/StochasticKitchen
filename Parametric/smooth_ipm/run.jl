# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with the Ipopt solver

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

# run optimization

include("ipopt.jl")

# model

model = PM.ParametricMultistageModel(
    states,
    controls,
    noises,
    dynamics,
    stage_cost,
    final_cost,
    horizon,
    zeros(horizon),
    zeros(2),
    stage_cost_gradient,
    final_cost_gradient)

set_processors(1)

# log
log = Dict(
    "x_opt" => prob.x,
    "f_values" => f_values,
    "n_steps" => n_steps,
    "overall_time" => overall_time,
    "average_time_per_gradient_call" => mean(output.elapsed_per_oracle_call[2:end]),
    "params" => params)

save("/home/StochasticKitchen/Parametric/results/ipopt/log.jld2", "log", log)