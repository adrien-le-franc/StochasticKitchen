# developed with Julia 1.1.1
#
# Interface for calibration and simulation
# of SDP models for EMS control


using Distributed

include("arguments.jl")


args = parse_commandline()

if args["workers"] > 1
	addprocs(args["workers"])
end

@eval @everywhere args=$args
@everywhere include(joinpath(@__DIR__, "models", args["model"]*".jl"))

if args["n_lags"] > 0
	@everywhere args["model"] = args["model"]*"_$(args["n_lags"])"
end

if args["calibrate"]
	println("\ncalibrate $(args["model"])\n")

	if args["workers"] > 1

		SDP.calibrate_sites_parallel(controller, 
                              joinpath(args["save"], args["model"]),
                              args["price"],
                              args["metadata"],
                              args["train"])

	else

		SDP.calibrate_sites(controller, 
		joinpath(args["save"], args["model"]),
		args["price"],
		args["metadata"],
		args["train"])

	end

end

if args["simulate"]
	println("\nsimulate $(args["model"])\n")

	EMSx.simulate_sites(controller, 
		joinpath(args["save"], args["model"]),
	    args["price"],
	    args["metadata"],
	    args["test"],
	    args["train"])

end