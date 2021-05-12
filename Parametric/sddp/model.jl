# developed with julia 1.5.3
#
# PV unit toy model with SDDP

using Distributed 

@everywhere begin 

	using SDDP, Clp #, CPLEX
	using ControlVariables
	const CV = ControlVariables

end

using JLD


# physical values

include("../parameters.jl")
include("../functions.jl")

# noises

pv = load(joinpath(@__DIR__, "..", "data", "ausgrid.jld"))["data"]*peak_power
weights, noises = CV.fit_linear_noise_model(collect(pv'), 10)
noises = clean_support(noises, 1.)

@eval @everywhere weights = $weights
@eval @everywhere noises = $noises

# model

compute_lower_bound(variable::Array{Float64,1}) = -sum(variable.*price*dt)

function parametric_sddp(variable::Array{Float64,1})

	model = SDDP.LinearPolicyGraph(
	    stages =  2*horizon,
	    sense = :Min,
	    lower_bound = compute_lower_bound(variable),
	    optimizer = Clp.Optimizer,) do subproblem, node

	    # solver verbose 

		set_silent(subproblem)

	    # State variables (normalized to [0, 1])
	    
	    @variable(subproblem, 0. <= soc <= max_battery_capacity, SDDP.State, initial_value = 0.)
	    @variable(subproblem, pv, SDDP.State, initial_value = 0.)
	    @variable(subproblem, parameter[i = 1:horizon], SDDP.State, initial_value = variable[i])


	    # Control variables (in DH)
	    
	    @variable(subproblem, 0. <= u_p <= max_battery_power*dt, SDDP.State, initial_value = 0.)
	    @variable(subproblem, 0. <= u_n <= max_battery_power*dt, SDDP.State, initial_value = 0.)

	    # Noise variable

	    @variable(subproblem, epsilon)

	    # other variables 

	    @variable(subproblem, lift) 
	    @expression(subproblem, delivery, pv.out*dt - (u_p.in - u_n.in))

	    # Transition functions and constraints

	    @constraint(subproblem, [i = 1:horizon], parameter[i].out == parameter[i].in)

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
	    	@constraint(subproblem, lift >= delivery - parameter[node÷2].in*dt)
	    	@constraint(subproblem, lift >= -delivery + parameter[node÷2].in*dt)
	        @stageobjective(subproblem, -price[node÷2]*parameter[node÷2].in*dt 
	        	+ penalty_coefficient*price[node÷2]*lift)
	    end

	    if (node == 2*horizon) # final node
	    	@stageobjective(subproblem, - price[node÷2]*parameter[node÷2].in*dt 
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

function compute_a_subgradient(variable::Array{Float64,1}, model::SDDP.PolicyGraph{Int64})

	value_function = SDDP.ValueFunction(model[1])
	set_silent(value_function.model)
	f, g = SDDP.evaluate(value_function,
		merge(Dict(:soc => 0., :u_p => 0., :u_n => 0., :pv => 0.),
			Dict(Symbol("parameter[$i]") => variable[i] for i in 1:horizon)))

	subgradient = [g[Symbol("parameter[$i]")] for i in 1:horizon]

	return subgradient, f

end