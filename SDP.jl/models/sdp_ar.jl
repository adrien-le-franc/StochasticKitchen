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

using LinearAlgebra
using DataFrames
using JLD


mutable struct SdpAR <: EMSx.AbstractController
   model::StoOpt.SDP 
end


## constant values and pointers for both simulation & calibration


const dx = 0.1
const du = 0.1
const horizon = 672
const n_lags = 1

const site_pointer = Ref(EMSx.Site("", EMSx.Battery(0., 0., 0., 0.), "", ""))
const price_pointer = Ref(EMSx.Price("", [0.], [0.]))
const net_demand_bounds_pointer = Ref(Dict("upper"=>0., "lower"=>0.))
const forecast_model_pointer = Ref(Dict("week_day"=>Array{Float64}(undef, 0, 0), 
    "week_end"=>Array{Float64}(undef, 0, 0)))

function apply_forecast(quarter::Int64, week_end::Bool, lags::Array{Float64,1})
    day_type = ["week_day", "week_end"][week_end+1]
    weights = forecast_model_pointer.x[day_type][quarter, :]
    push!(lags, 1.)
    return weights'*lags
end

function net_demand_dynamics(t::Int64, state::Array{Float64,1}, noise::Array{Float64,1})
    quarter = t%96 + (t%96 == 0)*96
    week_end = (t > 480)
    net_demand_forecast = apply_forecast(quarter, week_end, state[2:end])
    return min(1., max(0.,  net_demand_forecast + noise[1]))
end

function offline_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
    control = control[1]*site_pointer.x.battery.power*0.25
    normalized_net_demand_forecast = net_demand_dynamics(t, state, noise)
    net_demand_forecast = SDP.denormalize(normalized_net_demand_forecast,
        net_demand_bounds_pointer.x["upper"], 
        net_demand_bounds_pointer.x["lower"])
    imported_energy = control + net_demand_forecast
    return (price_pointer.x.buy[t]*max(0.,imported_energy) - 
        price_pointer.x.sell[t]*max(0.,-imported_energy))
end

function offline_dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
    noise::Array{Float64,1})
    scale_factor = site_pointer.x.battery.power*0.25/site_pointer.x.battery.capacity
    soc = state[1] + (site_pointer.x.battery.charge_efficiency*max(0.,control[1]) - 
        max(0.,-control[1])/site_pointer.x.battery.discharge_efficiency)*scale_factor
    net_demand_forecast = net_demand_dynamics(t, state, noise)
    return [soc, net_demand_forecast, state[2:end-1]...]
end

const model = StoOpt.SDP(Grid([0:dx:1 for i in 1:n_lags+1]..., enumerate=true), 
	Grid(-1:du:1), 
    nothing,
    offline_cost,
    offline_dynamics,
    horizon)

const controller = SdpAR(model)


## calibration specific functions


function calibrate_forecast(controller::SdpAR, lags_data::DataFrame)

    forecast_model = Dict{String, Array{Float64,2}}()

    for day_type in ["week_day", "week_end"]

        week_end = (day_type == "week_end")
        model_weights = Float64[]

        for quarter in 1:96

            data = lags_data[((lags_data.quarter .== quarter).*
                (lags_data.week_end .== week_end)), :]

            n_data = size(data, 1)
            observed = hcat(collect(hcat(data[:lags]...)'), ones(n_data))
            targets = data[:target]
            weights = pinv(observed'*observed)*observed'*targets

            append!(model_weights, weights)

        end

        forecast_model[day_type] = collect(reshape(model_weights, :, 96)')

    end

    return forecast_model
    
end



function SDP.update_site!(controller::SdpAR, site::EMSx.Site)

	site_pointer.x = site

    net_demand, upper_bound, lower_bound = SDP.normalized_net_demand(site.path_to_data_csv)
    net_demand_bounds_pointer.x = Dict("upper"=>upper_bound, "lower"=>lower_bound)

    lags_data = SDP.extract_lags(net_demand, n_lags)

    forecast_model_pointer.x = calibrate_forecast(controller, lags_data)

    save(joinpath(site.path_to_save_folder, 
        "forecast_model",
        site.id*".jld"), forecast_model_pointer.x)

    forecast_error = SDP.forecast_error_offline_law(lags_data, apply_forecast)

    controller.model.noises = SDP.data_frames_to_noises(forecast_error)

end

function SDP.update_price!(controller::SdpAR, price::EMSx.Price)
	price_pointer.x = price
end

function SDP.compute_value_functions(controller::SdpAR)
	return StoOpt.compute_value_functions(controller.model)
end


## simulation specific function and pointer


const value_functions_pointer = Ref(StoOpt.ArrayValueFunctions([0.]))

function load_value_functions(site_id::String, price_name::String)
    return load(joinpath(args["save"],
                args["model"], 
                "value_functions", 
                site_id*".jld"))["value_functions"][price_name]
end

function load_forecast_model(site_id::String)
    return load(joinpath(args["save"],
                args["model"], 
                "forecast_model", 
                site_id*".jld"))
end

function EMSx.compute_control(controller::SdpAR, information::EMSx.Information)
    
    if information.t == 1

        if site_pointer.x.id != information.site_id

            site_pointer.x = EMSx.Site(information.site_id, 
                                    information.battery, 
                                    joinpath(args["train"], information.site_id*".csv"), 
                                    args["save"]*args["model"])

            forecast_model_pointer.x = load_forecast_model(information.site_id)

            net_demand, upper_bound, lower_bound = SDP.normalized_net_demand(site_pointer.x.path_to_data_csv)
            net_demand_bounds_pointer.x = Dict("upper"=>upper_bound, "lower"=>lower_bound)
            lags_data = SDP.extract_lags(net_demand, n_lags)
            forecast_error = SDP.forecast_error_offline_law(lags_data, apply_forecast)
            controller.model.noises = SDP.data_frames_to_noises(forecast_error)

            value_functions_pointer.x = load_value_functions(information.site_id, 
                information.price.name)

        end

        if price_pointer.x.name != information.price.name
            price_pointer.x = information.price
            value_functions_pointer.x = load_value_functions(information.site_id, 
                information.price.name)
        end
    
    end

    net_demand_lags = information.load[1:n_lags] - information.pv[1:n_lags]
    net_demand_lags = SDP.normalize(net_demand_lags,
        net_demand_bounds_pointer.x["upper"], 
        net_demand_bounds_pointer.x["lower"])

    control = compute_control(controller.model, 
        information.t, 
        [information.soc, net_demand_lags...],
        StoOpt.RandomVariable(controller.model.noises, information.t),
        value_functions_pointer.x)

    return control[1]

end