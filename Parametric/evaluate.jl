# developed with julia 1.5.3
#
# PV unit toy model with SDDP

using SDDP, CPLEX
using JLD
using Statistics

include("functions.jl")
set_processors(4)

@everywhere begin 

	using SDDP, CPLEX
	using ControlVariables
	const CV = ControlVariables

end

# physical values

include("parameters.jl")

# noises

pv = load(joinpath(@__DIR__, "data", "ausgrid.jld"))["data"]*peak_power
weights, noises = CV.fit_linear_noise_model(collect(pv'), 10)
noises = clean_support(noises, 1.)

@eval @everywhere weights = $weights
@eval @everywhere noises = $noises

# model

compute_lower_bound(parameter::Array{Float64,1}) = -sum(parameter.*price*dt)

function evaluation_model(parameter::Array{Float64,1})

	model = SDDP.LinearPolicyGraph(
	    stages =  2*horizon,
	    sense = :Min,
	    lower_bound = compute_lower_bound(parameter),
	    optimizer = CPLEX.Optimizer,) do subproblem, node

	    # solver verbose 

		set_silent(subproblem)

	    # State variables (normalized to [0, 1])
	    
	    @variable(subproblem, 0. <= soc <= max_battery_capacity, SDDP.State, initial_value = 0.)
	    @variable(subproblem, pv, SDDP.State, initial_value = 0.)
	  
	    # Control variables (in DH)
	    
	    @variable(subproblem, 0. <= u_p <= max_battery_power*dt, SDDP.State, initial_value = 0.)
	    @variable(subproblem, 0. <= u_n <= max_battery_power*dt, SDDP.State, initial_value = 0.)

	    # Noise variable

	    @variable(subproblem, epsilon)

	    # other variables 

	    @variable(subproblem, lift) 
	    @expression(subproblem, delivery, pv.out*dt - (u_p.in - u_n.in))

	    # Transition functions and constraints

	    if (node % 2 == 1.) # decision node
	        @constraints(subproblem, begin
	            soc.out == soc.in
	            pv.out == pv.in    
	            soc.in + u_p.out*rho_c - u_n.out/rho_d >= 0.
	            soc.in + u_p.out*rho_c - u_n.out/rho_d <= max_battery_capacity
	        end)
	    end
	    
	    if (node % 2 == 0.) # noisy node
	        @constraints(subproblem, begin
	            u_p.out == u_p.in
	            u_n.out == u_n.in
	            pv.out == pv.in*weights[1, node÷2] + weights[2, node÷2] + epsilon
	            soc.out == soc.in + u_p.in*rho_c - u_n.in/rho_c 
	        end)
	    end

	    # Stage objective

	    if (node % 2 == 1.) # decision node
	        @stageobjective(subproblem, 0)
	    end

	    if (node % 2 == 0.) # noisy node
	    	@constraint(subproblem, lift >= delivery - parameter[node÷2]*dt)
	    	@constraint(subproblem, lift >= -delivery + parameter[node÷2]*dt)
	        @stageobjective(subproblem, -price[node÷2]*delivery 
	        	+ penalty_coefficient*price[node÷2]*lift)
	    end

	    if (node == 2*horizon) # final node
	    	@stageobjective(subproblem, - price[node÷2]*delivery 
	        	+ penalty_coefficient*price[node÷2]*lift - price[horizon]*soc.out)
	    end

	    # Noise

	    if (node % 2 == 0) # noisy node
	        SDDP.parameterize(subproblem, 
	        	reshape(noises[node÷2].value, :), 
	        	noises[node÷2].probability) do w
	            JuMP.fix(epsilon, w)
	        end
	    end

	end

	return model

end

function evaluate(profile::Array{Float64,1})

	model = evaluation_model(profile)
	SDDP.train(model, iteration_limit=100, parallel_scheme = SDDP.Asynchronous())
	n_samples = 10_000
	simulations = SDDP.simulate(model, n_samples; parallel_scheme = SDDP.Asynchronous())
	objective_values = [sum(stage[:stage_objective] for stage in sim) for sim in simulations]
	ub = round(mean(objective_values), digits = 2);
	lb = round(SDDP.calculate_bound(model), digits = 2)

	println("Simulation: ", ub, "- std :", std(objective_values))
	println("Lower bound: ", lb)

	return ub, lb

end

function process(path::String)

	for file in readdir(path)

		log = load(joinpath(path, file))["log"]
		p_opt = log["x_opt"] 
		ub, lb = evaluate(p_opt)
		log["ub"] = ub
		log["lb"] = lb
		save(joinpath(path, file), "log", log)

	end

end

process(joinpath(@__DIR__, "results", "sddp_psm"))

set_processors(1)