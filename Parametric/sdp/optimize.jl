# developed with julia 1.5.3


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
	
	
	if k % 1 == 0
		println("step $(k): $(cost_to_go)")
		#println("subgradient[24]: $(subgradient[:])")
		#println("variable[24]: $(variable[:])")
		#println("\n")
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