# developed with julia 1.5.3


using SubgradientMethods
SM = SubgradientMethods

using SDDP
using Dates


include("model.jl")


# oracle

struct SddpOracle <: SubgradientMethods.AbstractOracle end

function SubgradientMethods.call_oracle!(oracle::SddpOracle, 
	variable::Array{Float64,1}, k::Int64)

	model = parametric_sddp(variable)
	SDDP.train(model, time_limit=7.65, print_level=0, parallel_scheme = SDDP.Asynchronous())#, print_level=0)
	subgradient, cost_to_go = compute_a_subgradient(variable, model)

	if k % 1 == 0
		#println("step $(k): $(cost_to_go) - cuts : $(length(model.most_recent_training_results.log))")
		println(length(model.most_recent_training_results.log))
		#println("subgradient[24]: $(subgradient[24])")
		#println("variable[24]: $(variable[24])")
	end
	
	return cost_to_go, subgradient

end

# projection

struct HyperCubeProjection <: SubgradientMethods.AbstractProjection 
	project::Function
	coefficient::Float64
end

function HyperCubeProjection(coefficient::Float64)
	f(x::Array{Float64,1}) = max.(min.(coefficient, x), 0.)
	return HyperCubeProjection(f, coefficient) 
end

# stopping test

window = 5
epsilon = 0.005

function stop_progression(output::SM.Output)

	if length(output.all_values) < window
		return false
	else
		for i in length(output.all_values):-1:length(output.all_values)-window+1			
			if ((abs(output.all_values[i-1] - output.all_values[i])) 
				/ abs(output.all_values[i-1]) > epsilon)
				return false
			end
		end
		println("EXIT: Solved To Acceptable Level.")
		return true
	end
end

function SM.stopping_test(oracle::SddpOracle, output::SM.Output,
	parameters::SM.Parameters)

	if (output.elapsed > parameters.max_time || 
		output.subgradient_norm < parameters.epsilon ||
		stop_progression(output))
		return true
	else
		return false
	end

end