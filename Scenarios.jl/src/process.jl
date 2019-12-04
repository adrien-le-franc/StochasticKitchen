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

function calibrate_markov_error_process(df::DataFrame, steps::Array{Int64}, k::Int64)

	support = compute_support_of_error_process(df, steps, k)
	transitions = [zeros(k, k) for i in 1:length(steps)-1]
	initial_distributions = Dict("week_day"=>Dict(i=>Float64[] for i in 1:96), 
		"week_end"=>Dict(i=>Float64[] for i in 1:96))

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
	            day = day_type(is_week_end(date))
	            quarter = date_time_to_quarter(date)
	            push!(initial_distributions[day][quarter], delta)
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

	week_day_initial_distributions = InitialDistributions(initial_distributions["week_day"], k)
	week_end_initial_distributions = InitialDistributions(initial_distributions["week_end"], k)
	markov_chain = MarkovChain(support, transitions)

	generator = Dict("week_day"=>ScenarioGenerator(week_day_initial_distributions, markov_chain),
		"week_end"=>ScenarioGenerator(week_end_initial_distributions, markov_chain))

	return generator

end