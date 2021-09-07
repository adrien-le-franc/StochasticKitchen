# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with the Ipopt solver

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

# run optimization

include("ipopt.jl")
f(x) = prod(string.(split(string(x), ".")))

function run_all()

    for (dx, dg, du) in [(0.1, 0.1, 0.1), (0.1, 0.1, 0.05), (0.1, 0.05, 0.05), (0.05, 0.05, 0.05),
        (0.05, 0.05, 0.01), (0.05, 0.01, 0.01), (0.01, 0.01, 0.01)]

        states = PM.States(horizon, 0:dx:1., 0:dg:1.)
        controls = PM.Controls(horizon, -1:du:1)

        @eval @everywhere states = $states
        @eval @everywhere controls = $controls

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
        save("/home/StochasticKitchen/Parametric/results/smooth_ipm/log_$(f(dx))_$(f(dg))_$(f(du)).jld2", "log", log)

        println("dx = $(dx) ; dg = $(dg) ; du = $(du)")
        println(log)
        println("\n")

    end

end

run_all()

set_processors(1)