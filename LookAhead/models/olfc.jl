# developed with Julia 1.3.0
#
# Open Loop Feedback Controller (OLFC) 
# for EMS simulation
#
# Scenarios are computed with a package available at 
# https://github.com/adrien-le-franc/StochaticKitchen
#
# EMS simulation is performed with EMSx.jl 
# https://github.com/adrien-le-franc/EMSx.jl


using EMSx
using Scenarios

using JuMP, CPLEX
using DataFrames
using FileIO
using CSV

using MathOptInterface
const MOI = MathOptInterface


mutable struct Olfc <: EMSx.AbstractController
    model::Model
    scenario_generator::Dict{String, Scenarios.ScenarioGenerator}
    horizon::Int64
    n_scenarios::Int64
    Olfc() = new()
end


## constant values and functions for both simulation and calibration

const controller = Olfc()


function EMSx.initialize_site_controller(controller::Olfc, site::EMSx.Site)

    controller = Olfc()

    # scenario generation model
    steps = [1, 4, 8, 16, 48, 96]
    discrete_noise_points = 10
    scenario_generator = load_or_calibrate_scenario_generator(site, controller, steps, 
        discrete_noise_points)

    # LP model
    model = Model(with_optimizer(CPLEX.Optimizer))
    MOI.set(model, MOI.RawParameter("CPX_PARAM_SCRIND"), 0)

    horizon = 96
    n_scenarios = args["n_scenarios"]
    battery = site.battery

    @variable(model, 0 <= u_c[1:horizon])
    @variable(model, 0 <= u_d[1:horizon])
    @variable(model, 0 <= x[1:horizon+1])
    @variable(model, 0 <= z[1:horizon, 1:n_scenarios])
    @variable(model, w[1:horizon, 1:n_scenarios])
    @variable(model, x0)

    @expression(model, u,  u_c - u_d)

    @constraint(model, u_c .<= battery.power*0.25)
    @constraint(model, u_d .<= battery.power*0.25)
    @constraint(model, x .<= battery.capacity)
    @constraint(model, [k = 1:n_scenarios], u.+w[:, k] .<= z[:, k])
    @constraint(model, x[1] == x0)
    @constraint(model, dynamics, diff(x) .== u_c*battery.charge_efficiency .- 
        u_d/battery.discharge_efficiency)

    controller.model = model
    controller.scenario_generator = scenario_generator
    controller.horizon = horizon
    controller.n_scenarios = n_scenarios

    return controller
    
end

## calibration specific functions (no calibration here, done in simulation)

function load_or_calibrate_scenario_generator(site::EMSx.Site, 
    controller::EMSx.AbstractController, steps::Array{Int64,1}, k::Int64)
    
    #NB: same as ...forecast_model ?

    path_to_model = joinpath(site.path_to_save_folder, "scenario_generator", site.id*".jld2")

    if isfile(path_to_model)
        return load(path_to_model, "scenario_generator")
    else
        scenario_generator = calibrate_scenario_generator(site, controller, steps, k)
        EMSx.make_directory(joinpath(site.path_to_save_folder, "scenario_generator"))
        save(path_to_model, "scenario_generator", scenario_generator)
        return scenario_generator
    end

end

function calibrate_scenario_generator(site::EMSx.Site, 
    controller::EMSx.AbstractController, steps::Array{Int64,1}, k::Int64)

    train_data = CSV.read(site.path_to_train_data_csv)
    week_days_generator = Scenarios.ScenarioGenerator(Scenarios.week_days(train_data),
        steps, k)
    week_end_days_generator = Scenarios.ScenarioGenerator(Scenarios.week_end_days(train_data), 
        steps, k)

    scenario_generator = Dict("week_day"=>week_days_generator, 
        "week_end"=>week_end_days_generator)

    return scenario_generator
    
end


## simulation specific functions

function EMSx.update_price!(controller::Olfc, price::EMSx.Price)
    return nothing
end

function EMSx.compute_control(controller::Olfc, information::EMSx.Information)

    fix(controller.model[:x0], information.soc*information.battery.capacity)

    # generate scenarios
    probabilities = Float64[]
    forecast = information.forecast_load - information.forecast_pv
    quarter = information.t%96 + (information.t%96 == 0)*96
    day = ["week_day", "week_end"][(information.t > 480)+1]

    for k in 1:controller.n_scenarios
        scenario, probability = Scenarios.sample(controller.scenario_generator[day], quarter)
        fix.(controller.model[:w][:, k], forecast + scenario)
        push!(probabilities, probability)
    end

    probabilities = probabilities / sum(probabilities)
    
    # set prices, padding out of test period prices with zero values
    price = information.price
    price_window = information.t:min(information.t+controller.horizon-1, size(price.buy, 1))
    if length(price_window) != controller.horizon
        padding = controller.horizon - length(price_window)
        price = EMSx.Price(price.name, vcat(price.buy[price_window], zeros(padding)), 
            vcat(price.sell[price_window], zeros(padding)))
    else
        price = EMSx.Price(price.name, price.buy[price_window], price.sell[price_window])
    end

    @objective(controller.model, Min, 
        sum(probabilities .* [sum(price.buy.*controller.model[:z][:, k] -
            price.sell.*(controller.model[:z][:, k] - controller.model[:u] -
                controller.model[:w][:, k])) for k in 1:controller.n_scenarios]))

    optimize!(controller.model)

    return value(controller.model[:u][1]) / (information.battery.power*0.25)

end
