# developed with Julia 1.1.1
#
# functions for calibrating models for microgrid control
# models are  computed with a package available at 
# https://github.com/adrien-le-franc/StoOpt.jl


is_week_end(date::Union{Dates.Date, Dates.DateTime}) = Dates.dayofweek(date) in [6, 7]
day_type(week_end::Bool) = ["week_day", "week_end"][week_end+1]
date_time_to_quarter(timer::Dates.Time) = Int64(Dates.hour(timer)*4 + Dates.minute(timer)/15 + 1)
noise_data_df() = DataFrame(timestamp=Dates.Time(0, 0, 0):Dates.Minute(15):Dates.Time(23, 45, 0), 
        data=[Float64[] for i in 1:96])

function normalize(x::Array{Float64,1}, upper_bound::Float64, lower_bound::Float64) 
    return (x .- lower_bound) / (upper_bound - lower_bound)
end

function denormalize(x::Float64, upper_bound, lower_bound)
    return x*(upper_bound - lower_bound) + lower_bound
end


## Generic data parsing functions


function net_demand_offline_law(path_to_data_csv::String; k::Int64=10)

    """
    parse offline data to return a Dict of DataFrame objects with columns
    :timestamp -> Dates.Time ; one day discretized at 15 min steps
    :value -> Array{Float64,1} ; scalar values of a stochastic process
    :probability -> Arra{Float64,1} ; probabilities of each scalar value
    """

    data = CSV.read(path_to_data_csv)
    sorted_data = parse_data_frame(data)
    offline_law_data_frames = data_to_offline_law(sorted_data, k=k)

    return offline_law_data_frames
    
end

function parse_data_frame(data::DataFrame)

    train_data = Dict("week_day"=>noise_data_df(), "week_end"=>noise_data_df())

    for df in eachrow(data)

        timestamp = df[:timestamp]
        date = timestamp - Dates.Minute(15)
        day = Dates.Date(date)
        timing = Dates.Time(date)

        net_energy_demand = df[:actual_consumption] - df[:actual_pv]

        if is_week_end(day)
            push!(train_data["week_end"][date_time_to_quarter(timing), :data], 
                net_energy_demand)
        else
            push!(train_data["week_day"][date_time_to_quarter(timing), :data], 
                net_energy_demand)
        end

    end

    return train_data

end

function data_to_offline_law(data::Dict{String, DataFrame};
    k::Int64=10)

    offline_law_data_frames = Dict("week_day"=>DataFrame(), "week_end"=>DataFrame())

    for (key, df) in data

        law_df = DataFrame(timestamp=Dates.Time[], value=Array{Float64,1}[],
            probability=Array{Float64,1}[])

        for timestamp in Dates.Time(0, 0, 0):Dates.Minute(15):Dates.Time(23, 45, 0)

            noise_data = reshape(df[df.timestamp .== timestamp, :data][1], (1, :))
            n_data = length(noise_data)
            k_means = kmeans(noise_data, k)
            value = reshape(k_means.centers, :)
            probability = reshape(k_means.counts, :) / n_data
            push!(law_df, [timestamp, value, probability])

        end

        offline_law_data_frames[key] = law_df

    end

   return offline_law_data_frames

end


## lags parsing for forecast models


function normalized_net_demand(path_to_data_csv::String)

    data = CSV.read(path_to_data_csv)

    df = data[:, [:timestamp]]
    net_demand = (data[:actual_consumption] - data[:actual_pv])
    upper_bound = maximum(net_demand)
    lower_bound = minimum(net_demand)
    df[:net_demand] = (net_demand .- lower_bound) / (upper_bound - lower_bound)

    return df, upper_bound, lower_bound
end

function extract_lags(net_demand::DataFrame, n_lags::Int64)

    lags_df = DataFrame(week_end=Bool[], quarter=Int64[], lags=Array{Float64,1}[], 
        target=Float64[])

    for df in eachrow(net_demand)

        timestamp = df[:timestamp]
        target = df[:net_demand]
        date = timestamp - Dates.Minute(15) # 15 min shift for ahead of stamp data
        net_demand_lags = Float64[]

        for l in 1:n_lags
            lag_stamp = timestamp - Dates.Minute(15*l)
            if lag_stamp in net_demand[:timestamp]
                lag_value = net_demand[net_demand.timestamp .== lag_stamp, :net_demand][1]
                push!(net_demand_lags, lag_value) 
            else break
            end
        end

        if length(net_demand_lags) != n_lags
            continue
        end

        quarter = date_time_to_quarter(Dates.Time(date))
        push!(lags_df, [is_week_end(date), quarter, net_demand_lags, target])

    end
    
    return lags_df

end

function forecast_error_offline_law(lags_data::DataFrame, apply_forecast::Function)

    error_data_frame = Dict("week_day"=>noise_data_df(), "week_end"=>noise_data_df())

    for df in eachrow(lags_data)

        forecast = apply_forecast(df[:quarter], df[:week_end], df[:lags])
        forecast_error = df[:target] - forecast
        push!(error_data_frame[day_type(df[:week_end])][df[:quarter], :data], forecast_error)

    end

    offline_law_data_frames = data_to_offline_law(error_data_frame)

    return offline_law_data_frames

end

function load_or_calibrate_forecast_model(site::EMSx.Site, controller::EMSx.AbstractController, 
    lags_data::DataFrame)

    path_to_model = joinpath(site.path_to_save_folder, "forecast_model", site.id*".jld")

    if isfile(path_to_model)
        return load(path_to_model)
    else
        forecast_model = calibrate_forecast(controller, lags_data)
        EMSx.make_directory(joinpath(site.path_to_save_folder, "forecast_model"))
        save(path_to_model, forecast_model)
        return forecast_model
    end

end


## StoOpt specific data parsing function
## enables connecting the generic offline data pipeline
## with the StoOpt package


function data_frames_to_noises(offline_law::Dict{String,DataFrame})

    w_week_day = hcat(offline_law["week_day"][:value]...)'
    pw_week_day = hcat(offline_law["week_day"][:probability]...)'
    w_week_end = hcat(offline_law["week_end"][:value]...)'
    pw_week_end = hcat(offline_law["week_end"][:probability]...)'

    # one-week-long stochastic process
    w = vcat([w_week_day for i in 1:5]..., [w_week_end for i in 1:2]...)
    pw = vcat([pw_week_day for i in 1:5]..., [pw_week_end for i in 1:2]...)

    return StoOpt.Noises(w, pw)

end


### hackable functions


function calibrate_forecast(controller::EMSx.AbstractController, lags_data::DataFrame)
    """hackable function to calibrate forecast models"""
    return nothing
end

function compute_value_functions(controller::EMSx.AbstractController)
    """hackable function to compute value functions"""
    return nothing
end