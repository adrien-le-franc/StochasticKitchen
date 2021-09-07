# developed with julia 1.5.3
#
# PV unit toy model with MY regularized costs


using Distributed

@everywhere begin
  	
  	using ParametricMultistage
	PM = ParametricMultistage
	using ControlVariables
	const CV = ControlVariables

end

using JLD


# physical values

include("../parameters.jl")
include("../functions.jl")

# noises

pv = load(joinpath(@__DIR__, "..", "data", "ausgrid.jld"))["data"]
weights, noises = CV.fit_linear_noise_model(collect(pv'), 10)
noises = clean_support(noises, 0.001)

@eval @everywhere weights = $weights
@eval @everywhere noises = $noises

@everywhere begin

	# regularization

	const mu = 0.1

	# states

	const dx = 0.1
	const dg = 0.05
	const states = PM.States(horizon, 0:dx:1., 0:dg:1.)

	# controls

	const du = 0.05
	const controls = PM.Controls(horizon, -1:du:1)

	# dynamics

	function dynamics(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1})

		scale_factor = max_battery_power*dt/max_battery_capacity
		normalized_exchanged_power = rho_c*max(0., control[1]) - max(0., -control[1])/rho_d
	    soc = state[1] + normalized_exchanged_power*scale_factor

	    predicted_pv = min(max(weights[:, t]'*[state[2], 1.] + noise[1], 0.), 1.)

	    return [soc, predicted_pv]

	end

	# smoothed costs

	function stage_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1})

		power_production = min(max(weights[:, t]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
		power_delivery = power_production - control[1]*max_battery_power # in kW
		price_kW = price[t]*dt
	    
	    if parameter[t] < power_delivery - (penalty_coefficient+1)*price_kW*mu

	        return (penalty_coefficient*power_delivery - (penalty_coefficient+1)*parameter[t] 
	        	- (penalty_coefficient+1)^2*mu*price_kW/2)*price_kW

	    elseif parameter[t] > power_delivery + (penalty_coefficient-1)*price_kW*mu

	        return (-penalty_coefficient*power_delivery + (penalty_coefficient-1)*parameter[t] 
	        	- (penalty_coefficient-1)^2*mu*price_kW/2)*price_kW
	    
	    else
	        return -power_delivery*price_kW + (power_delivery-parameter[t])^2/(2*mu)
	    end
    
	end

	function final_cost(state::Array{Float64,1}, parameter::Array{Float64,1}) ### ???
		-price[horizon]*state[1]*max_battery_capacity 
	end

	# smoothed cost subgradients

	function stage_cost_gradient(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1}) 

		power_production = min(max(weights[:, t]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
		power_delivery = power_production - control[1]*max_battery_power # in kW
		price_kW = price[t]*dt
		gradient = zeros(horizon)
	    
	    if parameter[t] < power_delivery - (penalty_coefficient+1)*price_kW*mu
	        gradient[t] = -(penalty_coefficient+1)*price_kW
	    elseif parameter[t] > power_delivery + (penalty_coefficient-1)*price_kW*mu
	        gradient[t] = (penalty_coefficient-1)*price_kW
	    else
	        gradient[t] -(power_delivery-parameter[t])/mu
	    end

		return gradient

	end

	function final_cost_gradient(state::Array{Float64,1}, parameter::Array{Float64,1})
		return zeros(horizon)
	end

end

# model

model = PM.ParametricMultistageModel(
	states,
	controls,
	noises,
	dynamics,
	stage_cost,
	final_cost,
	horizon,
	zeros(horizon),
	zeros(2),
	stage_cost_gradient,
	final_cost_gradient)