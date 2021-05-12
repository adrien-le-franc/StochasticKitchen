# developed with julia 1.5.3
#
# some useful functions

using Distributed

using ControlVariables
const CV = ControlVariables

function set_processors(n::Integer = Sys.CPU_THREADS; kw...)
    if n < 1
        error("number of workers must be greater than 0")
    elseif n == 1 && workers() != [1]
        rmprocs(workers())
    elseif n > nworkers()
        p = addprocs(n - (nprocs() == 1 ? 0 : nworkers()); kw...)
    elseif n < nworkers()
        rmprocs(workers()[n + 1:end])
    end
    return workers()
end

function clean_support(noises::Noises, tol=0.001)

	h = length(noises)
	rv = Array{CV.RandomVariable,1}(undef, h)

	for t in 1:h

		if any(abs.(noises[t].value) .<= tol)

			support = Float64[]
			probability = Float64[]

			for (i, v) in enumerate(noises[t].value)
				if abs(v) > tol
					push!(support, v)
					push!(probability, noises[t].probability[i])
				end
			end

			push!(support, 0.)
			push!(probability, 1-sum(probability))

		else

			support = noises[t].value
			probability = noises[t].probability

		end

		rv[t] = CV.RandomVariable(support, probability)

	end

	return Noises(rv)

end