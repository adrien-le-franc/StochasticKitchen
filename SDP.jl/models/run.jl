# developed with Julia 1.1.1
#
# Interface for calibration and simulation
# of SDP models for EMS control



include("arguments.jl")


args = parse_commandline()

include(args["model"]*".jl")

if args["calibrate"]

	if args["workers"] > 1

		using Distributed
		addprocs(args["workers"])
		@everywhere using Pkg
		@everywhere Pkg.activate("../")
		@everywhere include(args["model"]*".jl")

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

	EMSx.simulate_sites(controller, 
		joinpath(args["save"], args["model"]),
	    args["price"],
	    args["metadata"],
	    args["test"])

end