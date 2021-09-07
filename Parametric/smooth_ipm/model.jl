# developed with julia 1.5.3
#
# PV unit toy model with smoothed costs


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

	const lambda = 2.

	function cost(delivered, committed, t)
	    return -price[t]*dt*(delivered - lambda*abs(delivered - committed))
	end

	function stage_cost(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1})

		power_production = min(max(weights[:, t]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
		delivered = power_production - control[1]*max_battery_power # in kW
		price_kW = price[t]*dt

		if abs(delivered - parameter[t]) > mu*lambda*price_kW
	        return cost(delivered, parameter[t], t) - ((lambda*price_kW)^2)*mu/2.
	    else
	        return -delivered*price_kW + (delivered-parameter[t])^2/(2*mu)
	    end

	end

	function final_cost(state::Array{Float64,1}, parameter::Array{Float64,1})
		-price[horizon]*state[1]*max_battery_capacity 
	end

	# smoothed cost subgradients

	function stage_cost_gradient(t::Int64, state::Array{Float64,1}, control::Array{Float64,1}, 
		noise::Array{Float64,1}, parameter::Array{Float64,1}) 

		power_production = min(max(weights[:, t]'*[state[2], 1.] + noise[1], 0.), 1.)*peak_power
		delivered = power_production - control[1]*max_battery_power # in kW
		price_kW = price[t]*dt
		gradient = zeros(horizon)

		if abs(delivered - parameter[t]) > mu*lambda*price_kW
			gradient[t] = sign(parameter[t] - delivered)*lambda*price_kW
	    else
	    	gradient[t] = (parameter[t] - delivered)/mu
	    end

		return gradient

	end

	function final_cost_gradient(state::Array{Float64,1}, parameter::Array{Float64,1})
		return zeros(horizon)
	end

end