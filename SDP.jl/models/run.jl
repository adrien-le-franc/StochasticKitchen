# developed with Julia 1.1.1
#
# Interface for calibration and simulation
# of SDP models for EMS control



include("arguments.jl")


args = parse_commandline()

include(args["model"]*".jl")

if args["calibrate"]

	SDP.calibrate_sites(controller, 
		args["save"],
		args["price"],
		args["metadata"],
		args["train"])
end

if args["simulate"]

	EMSx.simulate_sites(controller, 
		joinpath(args["save"], args["model"]*".jld"),
	    args["price"],
	    args["metadata"],
	    args["test"])

end