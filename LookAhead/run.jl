# developed with Julia 1.1.1
#
# Interface for calibration and simulation
# of SDP models for EMS control


using Distributed

include("arguments.jl")


args = parse_commandline()

if args["workers"] > 1
	addprocs(args["workers"])
	@eval @everywhere args=$args
end

@everywhere include(joinpath(@__DIR__, "models", args["model"]*".jl"))

if args["model"] == "olfc"
	args["model"] = "olfc_$(args["n_scenarios"])"
end

if args["workers"] > 1

		EMSx.simulate_sites_parallel(controller, 
			joinpath(args["save"], args["model"]),
		    args["price"],
		    args["metadata"],
		    args["test"],
		    args["train"])

	else

		EMSx.simulate_sites(controller, 
			joinpath(args["save"], args["model"]),
		    args["price"],
		    args["metadata"],
		    args["test"],
		    args["train"])

end