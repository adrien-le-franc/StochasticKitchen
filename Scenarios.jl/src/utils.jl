# developed with Julia 1.1.1

is_week_end(date::Union{Dates.Date, Dates.DateTime}) = Dates.dayofweek(date) in [6, 7]
day_type(week_end::Bool) = ["week_day", "week_end"][week_end+1]
date_time_to_quarter(timer::Union{Dates.Time, Dates.DateTime}) = Int64(Dates.hour(timer)*4 + Dates.minute(timer)/15 + 1)
closest(data::Array{Float64,1}, value::Float64) = findmin(abs.(data - ones(length(data))*value))[2]

function normalize_transition_matrix!(x::Array{Float64,2})
    n, m = size(x)
    for i in 1:n
        s = sum(x[i, :])
        if s == 0
            continue
        end
        x[i, :] = x[i, :] / s
    end
    return nothing
end