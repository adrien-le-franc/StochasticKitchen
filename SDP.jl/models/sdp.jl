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
	controller.model.noises = SDP.data_frames_to_noises(site.path_to_data_csv)
end

function SDP.update_price!(controller::Sdp, price::EMSx.Price)
	price_pointer.x = price
end

function SDP.compute_value_functions(sdp::Sdp)
	return StoOpt.compute_value_functions(sdp.model)
end


## simulation specific function and pointer


const value_functions_pointer = Ref(StoOpt.ArrayValueFunctions([0.]))

function EMSx.compute_control(sdp::Sdp, information::EMSx.Information)
    
    if information.t == 1

        if site_pointer.x.id != information.site_id
            site_pointer.x = EMSx.Site(information.site_id, information.battery, "", "")
            controller.model.noises = SDP.data_frames_to_noises(joinpath(args["train"], 
                information.site_id*".csv"))
            value_functions_pointer.x = load(joinpath(args["save"], 
                information.site_id*".jld"))["value_functions"][information.price.name]
        end

        if price_pointer.x.name != information.price.name
            price_pointer.x = information.price
            value_functions_pointer.x = load(joinpath(args["save"], 
                information.site_id*".jld"))["value_functions"][information.price.name]
        end
    
    end

    control = compute_control(sdp.model, information.t, [information.soc],
        StoOpt.RandomVariable(sdp.model.noises, information.t), value_functions_pointer.x)

    return control[1]

end