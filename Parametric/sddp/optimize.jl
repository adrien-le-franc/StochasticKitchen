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

	println(variable)

	model = parametric_sddp(variable)
	SDDP.train(model, iteration_limit=50)#, parallel_scheme = SDDP.Asynchronous())#, print_level=0)
	subgradient, cost_to_go = compute_a_subgradient(variable, model)

	if k % 1 == 0
		println("step $(k): $(cost_to_go)")
		println("subgradient[24]: $(subgradient[24])")
		println("variable[24]: $(variable[24])")
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
