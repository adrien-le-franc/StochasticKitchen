# developed with Julia 1.3.0
#
# functions to calibrate EMS control models

using EMSx
using JuMP, CPLEX
using ProgressMeter

using MathOptInterface
const MOI = MathOptInterface


mutable struct AnticipativeController <: EMSx.AbstractController
	controls::Array{Float64}
	AnticipativeController() = new()
end


const controller = AnticipativeController()


function eval_sites(controller::EMSx.AbstractController,
	path_to_save_folder::String, 
	path_to_price_folder::String, 
	path_to_metadata_csv_file::String, 
	path_to_test_data_folder::String)
	
	prices = EMSx.load_prices(path_to_price_folder)
	sites = EMSx.load_sites(path_to_metadata_csv_file, 
		path_to_test_data_folder, nothing, path_to_save_folder)

	@showprogress for site in sites
		
		eval_site(site, prices)
		
	end

	return nothing

end

function eval_site(site::EMSx.Site, prices::Array{EMSx.Price})

	test_data, site = EMSx.load_site_data(site)
	periods = unique(test_data[!, :period_id])
	simulations = EMSx.Simulation[]

	@showprogress for period_id in periods

		test_data_period = test_data[test_data.period_id .== period_id, :]
		period = EMSx.Period(string(period_id), test_data_period, site, EMSx.Simulation[])
		simulate_period!(controller, period, prices)
		append!(simulations, period.simulations)

	end

	EMSx.save_simulations(site, simulations)

	return nothing 

end

function simulate_period!(controller::EMSx.AbstractController, period::EMSx.Period, 
	prices::Array{EMSx.Price})

	for price in prices

		controller = initialize_anticipative_controller(controller, period, price)
		simulation = EMSx.simulate_scenario(controller, period, price)
		push!(period.simulations, simulation)

	end

	return nothing

end

function initialize_anticipative_controller(controller::AnticipativeController, 
	period::EMSx.Period, price::EMSx.Price)
	
	controller = AnticipativeController()

	model = Model(with_optimizer(CPLEX.Optimizer))
	MOI.set(model, MOI.RawParameter("CPX_PARAM_SCRIND"), 0)

	horizon = 672
	battery = period.site.battery

	@variable(model, 0 <= u_c[1:horizon])
	@variable(model, 0 <= u_d[1:horizon])
	@variable(model, 0 <= x[1:horizon+1])
	@variable(model, 0 <= z[1:horizon])
	@variable(model, w[1:horizon])
	@variable(model, x0)

	@expression(model, u,  u_c - u_d)

	@constraint(model, u_c .<= battery.power*0.25)
	@constraint(model, u_d .<= battery.power*0.25)
	@constraint(model, x .<= battery.capacity)
	@constraint(model, u.+w .<= z)
	@constraint(model, x[1] == x0)
	@constraint(model, dynamics, diff(x) .== u_c*battery.charge_efficiency .- 
		u_d/battery.discharge_efficiency)

	fix(model[:x0], 0.)

	data = period.data[98:end, :]
	pv = vcat(data[!, :actual_pv], data[end, :actual_pv])
	load = vcat(data[!, :actual_consumption], data[end, :actual_consumption])
	net_energy_demand = load-pv 

	fix.(model[:w], net_energy_demand)

	@objective(model, Min, sum(price.buy.*model[:z]-
			price.sell.*(model[:z]-model[:u]-model[:w])))

	optimize!(model)

    controller.controls = value.(model[:u]) / (battery.power*0.25)

	return controller

end

function EMSx.compute_control(controller::AnticipativeController, information::EMSx.Information)
	return controller.controls[information.t]
end

eval_sites(controller, 
	"/home/StochasticKitchen/LookAhead/results/anticipative", 
	"/home/StochasticKitchen/data/prices", 
	"/home/EMSx.jl/data/metadata.csv", 
	"/home/EMSx.jl/data/test")