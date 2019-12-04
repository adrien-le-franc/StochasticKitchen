# developed with Julia 1.1.1

struct InitialDistributions
	support::Array{Array{Float64,1}}
	distributions::Array{Categorical}
end

function InitialDistributions(data::Dict{Int64, Array{Float64,1}}, k::Int64)
	distributions = DiscreteNonParametric[]
	support = Array{Float64,1}[]
	for (key, val) in data
		r = kmeans(reshape(val, (1, :)), k)
		push!(support, [r.centers...])
		push!(distributions, Categorical([r.counts...] / length(val)))
	end
	return InitialDistributions(support, distributions)
end

function sample(initialize::InitialDistributions, t::Int64) 
	k = rand(initialize.distributions[t], 1)[1]
	value = initialize.support[t][k]
	proba = probs(initialize.distributions[t])[k]
	return value, proba
end


struct MarkovChain
	support::Array{Float64,2}
	probabilities::Array{Array{Categorical}}
end

function MarkovChain(support::Array{Float64,2}, matrices::Array{Array{Float64,2}})
	probabilities = Array{Categorical}[]
	for matrix in matrices
		laws = Categorical[]
		for row in eachrow(matrix)
			push!(laws, Categorical(row))
		end
		push!(probabilities, laws)
	end
	return MarkovChain(support, probabilities)
end

function sample(m::MarkovChain, x_0::Float64)
	state = closest(m.support[1, :], x_0)
    states = Float64[]
    probas = Float64[]
    
    for (t, transition) in enumerate(m.probabilities)
        k = rand(transition[state], 1)[1]
        push!(states, m.support[t+1, k])
        push!(probas, probs(transition[state])[k])
        state = k
    end
    
    return states, probas
end


struct ScenarioGenerator
	initial_distributions::InitialDistributions
	markov_chain::MarkovChain
end

function sample(generator::ScenarioGenerator, t::Int64)

	x_0, p_0 = sample(generator.initial_distributions, t)
	x, p = sample(generator.markov_chain, x_0)

	return [x_0, x...], p_0*prod(p)

end