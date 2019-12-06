include("../models/arguments.jl")
args = parse_commandline()
args["metadata"] = "/home/StochasticKitchen/data/sample.csv"
args["model"] = "sdp_rf"

#include("../models/x_sdp.jl")

#@everywhere using Pkg; Pkg.activate("."); include("models/x_sdp.jl")
# @elapsed SDP.calibrate_sites_parallel(controller, 
#                       joinpath(args["save"], args["model"]),
#                       args["price"],
#                       args["metadata"],
#                       args["train"])
