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

using JLD


mutable struct Sdp <: EMSx.AbstractController
   model::StoOpt.SDP 
end


## constant values and pointers for both simulation & calibration


const dx = 0.1
const du = 0.1
const horizon = 672

const site_pointer = Ref(EMSx.Site("", EMSx.Battery(0., 0., 0., 0.), "", ""))
const price_pointer = Ref(EMSx.Price("", [0.], [0.]))

function offline_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
    control = control[1]*site_pointer.x.battery.power*0.25
    imported_energy = control + noise[1]
    return (price_pointer.x.buy[t]*max(0.,imported_energy) - 
        price_pointer.x.sell[t]*max(0.,-imported_energy))
end

function offline_dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
    scale_factor = site_pointer.x.battery.power*0.25/site_pointer.x.battery.capacity
    soc = state + (site_pointer.x.battery.charge_efficiency*max.(0.,control) - 
        max.(0.,-control)/site_pointer.x.battery.discharge_efficiency)*scale_factor
    return soc
end

const sdp = StoOpt.SDP(Grid(0:dx:1, enumerate=true),
	Grid(-1:du:1), 
    nothing,
    offline_cost,
    offline_dynamics,
    horizon)

const controller = Sdp(sdp)


## calibration specific functions 


function SDP.update_site!(controller::Sdp, site::EMSx.Site)
	site_pointer.x = site
    offline_law_data_frames = SDP.net_demand_offline_law(site.path_to_data_csv)
	controller.model.noises = SDP.data_frames_to_noises(offline_law_data_frames)
end

function SDP.update_price!(controller::Sdp, price::EMSx.Price)
	price_pointer.x = price
end

function SDP.compute_value_functions(controller::Sdp)
	return StoOpt.compute_value_functions(controller.model)
end


## simulation specific functions and pointer


const value_functions_pointer = Ref(StoOpt.ArrayValueFunctions([0.]))

function load_value_functions(site_id::String, price_name::String)
    return load(joinpath(args["save"],
                args["model"], 
                "value_functions", 
                site_id*".jld"))["value_functions"][price_name]
end

function EMSx.compute_control(controller::Sdp, information::EMSx.Information)
    
    if information.t == 1

        if site_pointer.x.id != information.site_id

            SDP.update_site!(controller, EMSx.Site(information.site_id, 
                                    information.battery, 
                                    joinpath(args["train"], information.site_id*".csv"), 
                                    args["save"]*args["model"]))

            value_functions_pointer.x = load_value_functions(information.site_id, 
                information.price.name)

        end

        if price_pointer.x.name != information.price.name
            
            SDP.update_price!(controller, information.price)
            
            value_functions_pointer.x = load_value_functions(information.site_id, 
                information.price.name)

        end
    
    end

    control = compute_control(controller.model, information.t, [information.soc],
        StoOpt.RandomVariable(controller.model.noises, information.t), value_functions_pointer.x)

    return control[1]

end