
# Load packages
using Pkg

Pkg.activate("COVID_19")

ENV["CMDSTAN"] = expanduser("~/cmdstan/")

using Plots, StatsPlots
using CSV, DataFrames, HTTP
using Dates, Measures
using Turing, ReverseDiff, Memoization
using StanSample
using RCall


# make directories to hold images
mkdir("Images")
mkdir("Images/Alberta")
#mkdir("Images/Texas") # note. In original script is used to hold images of Texas model



# Function to pull quantiles
function get_qs(y_pre_fit::DataFrame, 
                y_est::DataFrame;
                ll::Float64 = 0.025,
                ul::Float64 = 0.975)
  
  fit_n = size(y_pre_fit, 2)
  est_n = size(y_est, 2)
  y_fit_m = [mean(y_pre_fit[:, i]) for i in 1:fit_n]
  y_fit_ll = [quantile(y_pre_fit[:, i], ll) for i in 1:fit_n]
  y_fit_ul = [quantile(y_pre_fit[:, i], ul) for i in 1:fit_n]
  
  # Next post period
  y_post_m = [mean(y_est[:, i]) for i in 1:est_n]
  y_post_ll = [quantile(y_est[:, i], ll) for i in 1:est_n]
  y_post_ul = [quantile(y_est[:, i], ul) for i in 1:est_n]

  # combine values into one vector each
  synth_m = vcat(y_fit_m, y_post_m)
  synth_ll = vcat(y_fit_ll, y_post_ll)
  synth_ul = vcat(y_fit_ul, y_post_ul)

  return synth_m, synth_ll, synth_ul

end




# Function to plot baseplot
function plot_base(title::String,
  ylabel::String,
  xlabel::String;
  title_position = :left,
  size = (800, 500),
  xrotation = 45,
  legend = :topleft,
  legend_foreground_color = nothing,
  bottom_margin = 17mm,
  left_margin = 5mm,
  tickfontsize = 12,
  titlefontsize = 16,
  legendfontsize = 10,
  guidefontsize = 14,
  grid = :off, 
  kwargs...)

   return plot(;title = title,
    ylabel = ylabel,
    xlabel = xlabel,
    title_position = title_position,
    size = size,
    xrotation = xrotation,
    legend = legend,
    legend_foreground_color = legend_foreground_color,
    bottom_margin = bottom_margin, left_margin = left_margin,
    tickfontsize = tickfontsize, titlefontsize = titlefontsize,
    legendfontsize = legendfontsize,
    guidefontsize = guidefontsize,
    grid = grid, kwargs...)

end






# Download data and prep it for later steps
# Load data for Canada
url_canada = "https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv"
df_canada = CSV.read(HTTP.get(url_canada).body, DataFrame)
rename!(df_canada, :prname => :Province_State,
                   :numtoday => :cases,
                   :numconf => :total_cases)
df_canada = df_canada[ in([1, 99]).(df_canada.pruid) .== false, :]



# Load data for US
url_cdc = "https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD"
df_cdc = CSV.read(HTTP.get(url_cdc).body, DataFrame; normalizenames = true)
rename!(df_cdc, :submission_date => :date,
                :new_case => :cases,
                :tot_cases => :total_cases)
df_cdc.date = replace.(df_cdc.date, "/" => "-")
df_cdc.date = Date.(df_cdc.date, dateformat"mm-dd-yyyy")


# get full state names and combine with CDC data
state_url = "https://github.com/jasonong/List-of-US-States/raw/master/states.csv"
state_names = CSV.read(HTTP.get(state_url).body, DataFrame; normalizenames = true)
rename!(state_names, :Abbreviation => :state,
                     :State => :Province_State)
df_cdc = leftjoin(df_cdc, state_names, on = :state)
dropmissing!(df_cdc, :Province_State)



# Load stringency index
url_stringency = "https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_withnotes_2021.csv"
df_stringency = CSV.read(HTTP.get(url_stringency).body, DataFrame; normalizenames = true)
dropmissing!(df_stringency, :RegionName)
filter!(row -> in(["Canada", "United States"]).(row.CountryName), df_stringency)

df_stringency.Date = Date.(string.(df_stringency.Date), dateformat"yyyymmdd")


# build df and filter by date
df = append!(df_canada[:, [:date, :Province_State, :total_cases]], df_cdc[:, [:date, :Province_State, :total_cases]])
start_date = Date(2021, 07, 01) - Day(98)
end_date = Date(2021, 07, 01) + Day(90)
filter!(row -> start_date .≤ row.date .≤ end_date, df)

# add in population
# add in population totals to compute cases per 100k
fn_usa = "Data/US population.csv"
fn_can = "Data/Canada population.csv"
pop_canada = CSV.read(fn_can, DataFrame; normalizenames = true,)
pop_usa = CSV.read(fn_usa, DataFrame; normalizenames = true,)
rename!(pop_usa, :State => :Province_State)
rename!(pop_canada, :Geography => :Province_State)
pop_canada.Province_State = replace.(pop_canada.Province_State, " 6" => "")

# clean and parse population
pop_canada.X2019 = replace.(pop_canada.X2019, "," => "")
pop_canada.X2019 = parse.(Float64, pop_canada.X2019)
pop_usa.X2019 = replace.(pop_usa.X2019, "," => "")
pop_usa.X2019 = parse.(Float64, pop_usa.X2019)

pop = append!(pop_canada[:, [:Province_State, :X2019]], pop_usa[:, [:Province_State, :X2019]])




# First, combine cases data and population data
# Then filter
date_list = unique(df.date)[2:end]
df.newdate = [join(["X", df.date[i]]) for i in 1:size(df, 1)]
df.newdate = replace.(df.newdate, "-" => "_")
df = unstack(df[:, [:newdate, :Province_State, :total_cases]], :newdate, :total_cases)
df = innerjoin(df, pop, on = :Province_State)



# get cases pre day
pop_scale = 100_000
y_raw = diff(Matrix{Float64}(df[df.Province_State .== "Alberta", 2:(end-1)]), dims = 2) ./ df[df.Province_State .== "Alberta", end] * pop_scale
X_raw =  diff(Matrix{Float64}(df[df.Province_State .!= "Alberta", 2:(end-1)]), dims = 2) ./ df[df.Province_State .!= "Alberta", end] * pop_scale




# Build 7-day moving averages
# First get containers
y = Vector{Float64}()
X = Matrix{Float64}(undef, (size(X_raw, 1), size(X_raw, 2) - 6))

# Calculate 7-day moving average
for j in 7:length(y_raw)
  push!(y, sum(y_raw[(j-6):j])/7)
  
  for i in 1:size(X_raw, 1)
    X[i, (j-6)] = sum(X_raw[i, (j-6):j]) / 7
  end

end


# Build first stan model
# get names of all provinces and states that have a higher stringency on July 1,
# which is the date AB "opened for summer"
filter!(row -> row.Date .== Date(2021, 07, 01), df_stringency)
ab = df_stringency.StringencyIndex[df_stringency.RegionName .== "Alberta"]

# get value of AB stringecy index on July 1, 2021
stringency_idx = df_stringency.RegionName[df_stringency.StringencyIndex .> ab]

# Remove states with zero reported cases
indx = [sum(X[i, :] .≤ 0) for i in 1:63]
X = X[indx .== 0, :]

# Get names of provinces and states with stringency 
# index who are still in the dataset
Province_State = df.Province_State[df.Province_State .!= "Alberta"]
Province_State = Province_State[indx .== 0]

# Keep states and provinces with stringency index greater than AB
Province_State_idx = in(stringency_idx).(Province_State)

# Remove states with stringency index greater than AB on July 1, 2021
X = X[Province_State_idx, :]



# Fit normal model
# Build pre and post values
trt_value = 91
y_pre = y[1:trt_value]
y_post = y[(trt_value+1):end]
X_pre = X[:, 1:trt_value]
X_post = X[:, (trt_value+1):end]

# Data for Stan
AB_data = Dict(
  
  "N_pre" => size(X_pre, 2), # Numbers of pre observations
  "N_post" => size(X_post, 2), # Number of post observations
  "p" => size(X_pre, 1), # Size of donor pool
  "y_pre" => log.(y_pre),
  "X_pre" => log.(X_pre),
  "X_post" => log.(X_post),
  
  "scale_alpha" => 1,
  "scale_g" => 1,
  
  "slab_scale" => 1,
  "slab_df" => 5,
  
  "nu_g" => 3,
  "nu_l" => 3
  
)


# Run model
rc_AB = stan_sample(sm; 
                    data = AB_data,
                    num_samples = 1000,
                    num_warmups = 1000,
                    num_chains = 4,
                    delta = 0.999,
                    max_depth = 15,
                    seed = 33294)


# check diagnostics and extract parameters
if success(rc_AB)
    diagnose(sm)
    params_out = DataFrame(read_samples(sm))
    gq = stan_generate_quantities(sm)
end
  
# Transform parameters
y_fit = gq[:, 1:length(y_pre)]
y_post_est = gq[:, (length(y_pre) + 1):(length(y_pre) + length(y_post))]
y_fit = exp.(y_fit)
y_post_est = exp.(y_post_est)


# Build series for plotting.
# Start with pre-period fitted values
AB_m, AB_ll, AB_ul = get_qs(y_fit, y_post_est)

# dates
date_list = [(start_date + Day(7)):Day(1):end_date;]


AB_trend = plot_base("Synthetic control estimates: 7-day moving average",
                     "Num cases (per 100k)",
                     "Date";
                     legend = :bottomleft)
plot!(AB_trend, date_list, AB_m, 
      ribbon = (AB_m - AB_ll, AB_ul - AB_m),
      lc = :red, fill = :red, lw = 3,
      label = "Estimated trend")
plot!(AB_trend, date_list, vcat(y_pre, y_post), 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
annotate!(AB_trend, Date(2021, 07, 03), 45, 
          StatsPlots.text("Most restrictions dropped",
          10, :left))
png(AB_trend, "Images/Alberta/synthetic_control_estimates")
  
AB_trend




# Build counts by adding in values
cases_start = 3420.15

# Transform parameters
y_fit = gq[:, 1:length(y_pre)]
y_post_est = gq[:, (length(y_pre) + 1):(length(y_pre) + length(y_post))]
y_fit = exp.(y_fit)
y_post_est = exp.(y_post_est)
y_fit[:, 1] = y_fit[:, 1] .+ cases_start

for m in 2:size(y_fit, 2)
  for n in 1:size(y_fit, 1)
    y_fit[n, m] = sum(y_fit[n, (m-1):m])
  end
end

y_post_est[:, 1] = y_fit[:, end] + y_post_est[:, 1]

for m in 2:size(y_post_est, 2)
  for n in 1:size(y_post_est, 1)
    y_post_est[n, m] = sum(y_post_est[n, (m-1):m])
  end
end

# Build series for plotting.
# Start with pre-period fitted values
AB_m, AB_ll, AB_ul = get_qs(y_fit, y_post_est)


AB_trend = plot_base("Synthetic control estimates: total cases",
                     "Num cases (per 100k)",
                     "Date";
                     legend = :bottomleft)
plot!(AB_trend, date_list, AB_m, 
      ribbon = (AB_m - AB_ll, AB_ul - AB_m),
      lc = :red, fill = :red, lw = 3,
      label = "Estimated trend")
plot!(AB_trend, date_list, Vector(y_actual[:, 1]), 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
annotate!(AB_trend, Date(2021, 07, 03), 45, 
          StatsPlots.text("Most restrictions dropped",
          10, :left))
png(AB_trend, "Images/Alberta/synthetic_control_estimates")
  
AB_trend



#total number of cases
y_direct = Matrix{Float64}(df[df.Province_State .== "Alberta", 9:(end-1)]) ./ df[df.Province_State .== "Alberta", end] * pop_scale
X_direct = Matrix{Float64}(df[df.Province_State .!= "Alberta", 9:(end-1)]) ./ df[df.Province_State .!= "Alberta", end] * pop_scale

X_direct = X_direct[indx .== 0, :]
X_direct = X_direct[Province_State_idx, :]

y_direct = log.(y_direct)
X_direct = log.(X_direct)

# Fit normal model
# Build pre and post values
y_pre_direct = y_direct[1:trt_value]
y_post_direct = y_direct[(trt_value+1):end]
X_pre_direct = X_direct[:, 1:trt_value]
X_post_direct = X_direct[:, (trt_value+1):end]

y_mean_direct = y_pre_direct[end]
x_mean_direct = X_pre_direct[:, end]
#x_mean = [mean(X_pre[i, :]) for i in 1:size(X_pre, 1)]
y_sd = std(y_pre_direct)
x_sd = [std(X_pre_direct[i, :]) for i in 1:size(X_pre_direct, 1)]

y_pre_direct = (y_pre_direct .- y_mean_direct) #./ y_sd
X_pre_direct = (X_pre_direct .- x_mean_direct) #./ x_sd
y_post_direct = (y_post_direct .- y_mean_direct) #./ y_sd
X_post_direct = (X_post_direct .- x_mean_direct) #./ x_sd



AB_data_direct = Dict(
  
  "N_pre" => size(X_pre_direct, 2), # Numbers of pre observations
  "N_post" => size(X_post_direct, 2), # Number of post observations
  "p" => size(X_pre_direct, 1), # Size of donor pool
  "y_pre" => y_pre_direct,
  "X_pre" => X_pre_direct,
  "X_post" => X_post_direct,
  
  "scale_alpha" => 1,
  "scale_g" => 1,
  
  "slab_scale" => 1,
  "slab_df" => 5,
  
  "nu_g" => 3,
  "nu_l" => 3
  
)


# Run model
rc_AB_direct = stan_sample(sm; 
                           data = AB_data_direct,
                           num_samples = 1000,
                           num_warmups = 1000,
                           num_chains = 4,
                           delta = 0.999,
                           max_depth = 20,
                           seed = 33294)


# check diagnostics and extract parameters
if success(rc_AB_direct)
    diagnose(sm)
    params_out = DataFrame(read_samples(sm))
    gq = stan_generate_quantities(sm)
end

y_fit_direct = gq[:, 1:length(y_pre_direct)] .+ y_mean
y_post_est_direct = gq[:, (length(y_pre_direct) + 1):(length(y_pre_direct) + length(y_post_direct))] .+ y_mean
y_fit_direct = exp.(y_fit_direct)
y_post_est_direct = exp.(y_post_est_direct)
AB_m_direct, AB_ll_direct, AB_ul_direct = get_qs(y_fit_direct, y_post_est_direct)

y_actual = Matrix{Float64}(df[df.Province_State .== "Alberta", 9:(end-1)]) ./ df[df.Province_State .== "Alberta", end] * pop_scale

date_list = [(start_date):Day(1):end_date;]


AB_trend_direct = plot_base("Synthetic control estimates: Directly modelling\ncumulative cases",
                            "Num cases (per 100k)",
                            "Date")
plot!(AB_trend_direct, date_list, AB_m, 
      ribbon = (AB_m - AB_ll, AB_ul - AB_m),
      lc = :red, fill = :red, lw = 3,
      label = "Estimated trend";
      top_margin = 5mm)
plot!(AB_trend_direct, date_list, Vector(y_actual[1, :]), 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend_direct, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
hline!(AB_trend_direct, [exp(y_mean)],
       lc = :black, linestyle = :dash,
       linealpha = 0.5,
       lw = 2,
       label = "")
annotate!(AB_trend_direct, Date(2021, 07, 03), 45, 
          StatsPlots.text("Most restrictions dropped",
          10, :left))
png(AB_trend_direct, "Images/Alberta/cumulative_cases")








