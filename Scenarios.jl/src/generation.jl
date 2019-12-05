# developed with Julia 1.1.1


function compute_support_of_error_process(df::DataFrame, steps::Array{Int64}, k::Int64)

	errors = Dict(s=>Float64[] for s in steps)

	for (n, row) in enumerate(eachrow(df))

		if n == size(df, 1) - 96
			break
		end

		for s in steps
			forecast = row[5+s] - row[101+s] # demand - pv
			value = df[n+s, :actual_consumption] - df[n+s, :actual_pv]
			push!(errors[s], value - forecast)
		end 

	end

	support = zeros(length(steps), k)

	for (n, s) in enumerate(steps)

		c = kmeans(reshape(errors[s], (1, :)), k).centers
		c = sort!([c...])
		support[n, :] = c

	end

	return support

end


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
			push!(laws, Categorical(collect(row)))
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
	steps::Array{Int64,1}
end

function ScenarioGenerator(df::DataFrame, steps::Array{Int64,1}, k::Int64)

	support = compute_support_of_error_process(df, steps, k)
	transitions = [zeros(k, k) for i in 1:length(steps)-1]
	initial_distributions = Dict(i=>Float64[] for i in 1:96)

	for (n, row) in enumerate(eachrow(df))

		if n == size(df, 1) - 96
			break
		end

		states = Int64[]

		for (i, s) in enumerate(steps)

			forecast = row[5+s] - row[101+s]
	        value = df[n+s, :actual_consumption] - df[n+s, :actual_pv]
	        delta = value - forecast

	        #fill initial distributions
	        if s == 1
	            date = row[:timestamp]
	            quarter = date_time_to_quarter(date)
	            push!(initial_distributions[quarter], delta)
	        end

	        #record trajectory
	        push!(states, closest(support[i, :], delta))

		end

		#fill transition matrices
		for i in 1:length(states)-1
	        transitions[i][states[i], states[i+1]] += 1
	    end

	end

	for x in transitions
		normalize_transition_matrix!(x)
	end

	initial_distributions = InitialDistributions(initial_distributions, k)
	markov_chain = MarkovChain(support, transitions)

	return ScenarioGenerator(initial_distributions, markov_chain, steps)

end

function sample(generator::ScenarioGenerator, t::Int64)

	x_0, p_0 = sample(generator.initial_distributions, t)
	x, p = sample(generator.markov_chain, x_0)
	interpolator = interpolate((generator.steps, ), [x_0, x...], Gridded(Linear()))

	return [interpolator(i) for i in 1:96], p_0*prod(p)

end