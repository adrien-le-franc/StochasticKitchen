# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with the Ipopt solver

DIR = @__DIR__

include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include("model.jl")
include(joinpath(DIR, "optimize.jl"))
using Statistics

step_size(k::Int64) = 1000/k
const projection = HyperCubeProjection(peak_power)
const parameters = SubgradientMethods.Parameters(zeros(48), step_size, 100, Dates.Day(1), 0.01)
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

        oracle = ParametricMultistageOracle(model)
        output = SubgradientMethods.optimize!(oracle, projection, parameters)

        # log
        log = Dict(
            "x_opt" => output.variable,
            "f_values" => output.all_values,
            "n_steps" => output.final_iteration,
            "overall_time" => round(output.elapsed, Dates.Second),
            "average_time_per_gradient_call" => mean(output.elapsed_per_oracle_call[2:end]))

        save("/home/StochasticKitchen/Parametric/results/smooth_pgd/log_$(f(dx))_$(f(dg))_$(f(du)).jld2", "log", log)

        println("dx = $(dx) ; dg = $(dg) ; du = $(du)")
        println(log)
        println("\n")

    end

end

run_all()

set_processors(1)