# developed with julia 1.5.3
#
# Gradient descent method


DIR = @__DIR__
include(joinpath(DIR, "..", "functions.jl"))


using SubgradientMethods
SM = SubgradientMethods
using ParametricMultistage
PM = ParametricMultistage

using Dates


include("model.jl")


# oracle

mutable struct ParametricMultistageOracle <: SubgradientMethods.AbstractOracle
	model::ParametricMultistage.ParametricMultistageModel
end

function set_variable!(oracle::ParametricMultistageOracle, variable::Array{Float64,1})
	oracle.model.parameter = variable
end

function SubgradientMethods.call_oracle!(oracle::ParametricMultistageOracle, 
	variable::Array{Float64,1}, k::Int64)

	set_variable!(oracle, variable)
	subgradient, cost_to_go = PM.parallel_compute_gradient(oracle.model, true)	
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

function SM.stopping_test(oracle::ParametricMultistageOracle, output::SM.Output,
	parameters::SM.Parameters)

	if (output.elapsed > parameters.max_time || 
		output.subgradient_norm < parameters.epsilon ||
		stop_progression(output))
		return true
	else
		return false
	end

end
