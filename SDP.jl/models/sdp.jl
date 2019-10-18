# developed with Julia 1.1.1
#
# Stochastic Dynamic Programming (SDP) controller 
# for EMS simulation
#
# SDP is computed with a package available at 
# https://github.com/adrien-le-franc/StoOpt.jl
#
# EMS simulation is performed with EMSx.jl 
# https://github.com/adrien-le-franc/EMSx.jl


using SDP
using EMSx
using StoOpt

using FileIO


mutable struct Sdp <: EMSx.AbstractController
   model::StoOpt.SDP
   price::EMSx.Price
   value_functions::Union{StoOpt.ArrayValueFunctions, Nothing}
   Sdp() = new()
end


## constant values and functions for both simulation and calibration

const controller = Sdp()
const dx = 0.1
const du = 0.1
const horizon = 672 

function EMSx.initialize_site_controller(controller::Sdp, site::EMSx.Site)

    controller = Sdp()

    offline_law_data_frames = SDP.net_demand_offline_law(site.path_to_data_csv)
    noises = SDP.data_frames_to_noises(offline_law_data_frames)
    
    function offline_dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
        scale_factor = site.battery.power*0.25/site.battery.capacity
        soc = state + (site.battery.charge_efficiency*max.(0.,control) - 
            max.(0.,-control)/site.battery.discharge_efficiency)*scale_factor
        return soc
    end

    function offline_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
        control = control[1]*site.battery.power*0.25
        imported_energy = control + noise[1]
        return (controller.price.buy[t]*max(0.,imported_energy) - 
            controller.price.sell[t]*max(0.,-imported_energy))
    end

    model = StoOpt.SDP(Grid(0:dx:1, enumerate=true),
        Grid(-1:du:1), 
        noises,
        offline_cost,
        offline_dynamics,
        horizon)

    controller.model = model

   return controller
    
end

function EMSx.update_price!(controller::Sdp, price::EMSx.Price)
    controller.price = price
end

## calibration specific function

function SDP.compute_value_functions(controller::Sdp)
	return StoOpt.compute_value_functions(controller.model)
end

## simulation specific functions

function load_value_functions(site_id::String, price_name::String)
    return load(joinpath(args["save"],
                args["model"], 
                "value_functions", 
                site_id*".jld2"))["value_functions"][price_name]
end

function EMSx.compute_control(controller::Sdp, information::EMSx.Information)
    
    if information.t == 1
        controller.value_functions = load_value_functions(information.site_id, 
                information.price.name)
    end

    control = compute_control(controller.model, information.t, [information.soc],
        StoOpt.RandomVariable(controller.model.noises, information.t), controller.value_functions)

    return control[1]

end