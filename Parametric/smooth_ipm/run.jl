# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with the Ipopt solver

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include("ipopt.jl")

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

log = optimize(model)
save("/home/StochasticKitchen/Parametric/results/ipopt/log.jld2", "log", log)

set_processors(1)
