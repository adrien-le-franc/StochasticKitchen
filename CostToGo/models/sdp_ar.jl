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
using FileIO


mutable struct SdpAR <: EMSx.AbstractController
   model::StoOpt.SDP
   price::EMSx.Price
   upper_bound::Float64
   lower_bound::Float64
   forecast_model::Dict{String, Array{Float64,2}}
   value_functions::Union{StoOpt.ArrayValueFunctions, Nothing}
   SdpAR() = new()
end


## constant values and functions for both simulation and calibration

const controller = SdpAR()
const dx = 0.1
const du = 0.1
const horizon = 672
const n_lags = args["n_lags"]

function EMSx.initialize_site_controller(controller::SdpAR, site::EMSx.Site)

    controller = SdpAR()

    # forecast model 
    net_demand, upper_bound, lower_bound = SDP.normalized_net_demand(site.path_to_train_data_csv)
    lags_data = SDP.extract_lags(net_demand, n_lags)
    forecast_model = SDP.load_or_calibrate_forecast_model(site, controller, lags_data)

    # forecast error 
    function apply_forecast(quarter::Int64, week_end::Bool, lags::Array{Float64,1})
        day_type = ["week_day", "week_end"][week_end+1]
        weights = forecast_model[day_type][quarter, :]
        push!(lags, 1.)
        return weights'*lags
    end

    forecast_error = SDP.forecast_error_offline_law(lags_data, apply_forecast)
    noises = SDP.data_frames_to_noises(forecast_error)

    # dyamics and cost
    function net_demand_dynamics(t::Int64, state::Array{Float64,1}, noise::Array{Float64,1})
        quarter = t%96 + (t%96 == 0)*96
        week_end = (t > 480)
        net_demand_forecast = apply_forecast(quarter, week_end, state[2:end])
        return min(1., max(0.,  net_demand_forecast + noise[1]))
    end

    function offline_dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
        noise::Array{Float64,1})
        scale_factor = site.battery.power*0.25/site.battery.capacity
        soc = state[1] + (site.battery.charge_efficiency*max(0.,control[1]) - 
            max(0.,-control[1])/site.battery.discharge_efficiency)*scale_factor
        net_demand_forecast = net_demand_dynamics(t, state, noise)
        return [soc, net_demand_forecast, state[2:end-1]...]
    end

    function offline_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
        noise::Array{Float64,1})
        control = control[1]*site.battery.power*0.25
        normalized_net_demand_forecast = net_demand_dynamics(t, state, noise)
        net_demand_forecast = SDP.denormalize(normalized_net_demand_forecast,
            upper_bound, lower_bound)
        imported_energy = control + net_demand_forecast
        return (controller.price.buy[t]*max(0.,imported_energy) - 
            controller.price.sell[t]*max(0.,-imported_energy))
    end

    model = StoOpt.SDP(Grid([0:dx:1 for i in 1:n_lags+1]..., enumerate=true),
        Grid(-1:du:1), 
        noises,
        offline_cost,
        offline_dynamics,
        horizon)

    controller.model = model
    controller.forecast_model = forecast_model
    controller.upper_bound = upper_bound
    controller.lower_bound = lower_bound

   return controller
    
end

function EMSx.update_price!(controller::SdpAR, price::EMSx.Price)
    controller.price = price
end

## calibration specific functions

function SDP.calibrate_forecast(controller::SdpAR, lags_data::DataFrame)

    forecast_model = Dict{String, Array{Float64,2}}()

    for day_type in ["week_day", "week_end"]

        week_end = (day_type == "week_end")
        model_weights = Float64[]

        for quarter in 1:96

            data = lags_data[((lags_data.quarter .== quarter).*
                (lags_data.week_end .== week_end)), :]

            n_data = size(data, 1)
            observed = hcat(collect(hcat(data[!, :lags]...)'), ones(n_data))
            targets = data[!, :target]
            weights = pinv(observed'*observed)*observed'*targets

            append!(model_weights, weights)

        end

        forecast_model[day_type] = collect(reshape(model_weights, :, 96)')

    end

    return forecast_model
    
end

function SDP.compute_value_functions(controller::SdpAR)
	return StoOpt.compute_value_functions(controller.model)
end

## simulation specific functions

function load_value_functions(site_id::String, price_name::String)
    return load(joinpath(args["save"],
                args["model"], 
                "value_functions", 
                site_id*".jld2"))["value_functions"][price_name]
end

function EMSx.compute_control(controller::SdpAR, information::EMSx.Information)
    
    if information.t == 1
        controller.value_functions = load_value_functions(information.site_id, 
                information.price.name)
    end

    net_demand_lags = information.load[1:n_lags] - information.pv[1:n_lags]
    net_demand_lags = max.(0., min.(1., SDP.normalize(net_demand_lags, controller.upper_bound, 
        controller.lower_bound)))

    control = compute_control(controller.model, 
        information.t, 
        [information.soc, net_demand_lags...],
        StoOpt.RandomVariable(controller.model.noises, information.t),
        controller.value_functions)

    if control[1] > 1.
        println("t $(information.t), state $([information.soc, net_demand_lags...])")
    end

    return control[1]

end