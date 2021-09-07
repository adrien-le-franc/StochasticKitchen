# developed with Julia 1.5.3
#
# Day-ahead commitment profile optimization with SDDP

DIR = @__DIR__

using Statistics
include(joinpath(DIR, "..", "functions.jl"))

set_processors(4)

include(joinpath(DIR, "optimize.jl"))

step_size(k::Int64) = 1000/k
const projection = HyperCubeProjection(peak_power)
const parameters = SubgradientMethods.Parameters(zeros(48), step_size, 100, Dates.Day(1), 0.01)

function run_all()

    for n_cuts in [5, 10, 20, 40, 80, 150, 250, 500]

        oracle = SddpOracle(n_cuts)

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

        save("/home/StochasticKitchen/Parametric/results/sddp_psm/log_k$(n_cuts).jld2", "log", log)

        println("n_cuts = $(n_cuts) :")
        println(log)
        println("\n")

    end

end

run_all()

set_processors(1)