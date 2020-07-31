# developed with Julia 1.1.1
#
# arguments parsing for examples


using ArgParse


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin
        
        "--save"
            help = "folder to save scores and/or models"
            arg_type = String
            default = joinpath(@__DIR__, "results")

        "--metadata"
            help = "metadata.csv - site and battery parameters"     
            arg_type = String
            default = joinpath(@__DIR__, "../data/sample.csv")

        "--train"
            help = "train data folder"
            arg_type = String
            default = joinpath(@__DIR__, "../data/train")

        "--test"
            help = "test data folder"
            arg_type = String
            default = joinpath(@__DIR__, "../data/test")

        "--price"
            help = "price folder or .csv file"
            arg_type = String
            default = joinpath(@__DIR__, "../data/prices")

        "--model"
            help = "model for simulation"
            arg_type = String
            range_tester = in(["mpc", "olfc"])

        "--n_scenarios"
            help = "number of scenario for olfc"
            arg_type = Int64
            default = 10

        "--workers"
            help = "number of workers for parallelized calibration"
            arg_type = Int64
            default = 1  

    end

    parsed_args = parse_args(s)

    println("")
    for (key, val) in parsed_args
        println("  $key  =>  $(repr(val))")
    end
    println("")

    return parse_args(s)
    
end