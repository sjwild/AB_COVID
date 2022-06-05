# Load packages
ENV["CMDSTAN"] = expanduser("~/cmdstan/")
using Pkg

Pkg.activate("COVID_19")

using Plots, StatsPlots
using CSV, DataFrames, HTTP
using Dates, Measures
using Turing, ReverseDiff, Memoization
using StanSample
using RCall

# Note that ReadStatt.jl does not seem to work with Julia 1.7.2 yet. 
# So we will use RCall
R"""
library(tidyverse)
library(haven)

# This code comes from Scot Cunningham's Mixtape
read_data <- function(df)
{
  full_path <- paste("https://raw.github.com/scunning1975/mixtape/master/", 
                     df, sep = "")
  df <- read_dta(full_path)
  return(df)
}

texas <- read_data("texas.dta") %>%
  as.data.frame(.)

"""

df_texas = @rget texas





#### Code chunk 2 ####
drop_fips = [3, 7, 14, 43, 52] # States to drop


# Drop states
df_texas_bscm = df_texas[in(drop_fips).(df_texas.statefip) .== false, [:year, :statefip, :bmprison]]


# create name for state fips, which will be used as variable names when we go from long to widen format
df_texas_bscm.statefip = [join(["statefips_", df_texas_bscm.statefip[i]]) for i in 1:size(df_texas_bscm, 1)]
df_texas_bscm = unstack(df_texas_bscm, :statefip, :bmprison) # equivalent to pivot_wider


# build a vector and a matrix, than scale to help the sampler.
adj_factor = 10_000
ind = contains.(names(df_texas_bscm), "statefips_48") # index IDing texas
X_texas = Matrix{Float64}(df_texas_bscm[:, Not(ind)])[:, 2:51] ./ adj_factor
y_texas = Vector{Float64}(df_texas_bscm[:, ind][:, 1]) ./ adj_factor


# Matrices and vectors, divided into pre and post period
yr = 1993
X_texas_pre = X_texas[df_texas_bscm.year .≤ yr, :]
X_texas_post = X_texas[df_texas_bscm.year .> yr, :]
y_texas_pre = y_texas[df_texas_bscm.year .≤ yr]
y_texas_post = y_texas[df_texas_bscm.year .> yr]




#### Code chunk 3 ####
# run stan model
# Note we need to transpose our X matrices
texas_data = Dict(
  
  "N_pre" => size(X_texas_pre', 2), # Numbers of pre observations
  "N_post" => size(X_texas_post', 2), # Number of post observations
  "p" => size(X_texas_pre', 1), # Size of donor pool
  "y_pre" => y_texas_pre,
  "X_pre" => X_texas_pre',
  "X_post" => X_texas_post',
  
  "scale_alpha" => 1,
  "scale_g" => 1,
  
  "slab_scale" => 1,
  "slab_df" => 5,
  
  "nu_g" => 3,
  "nu_l" => 3
  
)

# load temp directory for model output
tmpdir = joinpath(@__DIR__, "tmp")

# run model
sm = SampleModel("bscm", read("Stan/bscm_horseshoe_modified.stan", String), tmpdir)
rc_texas = stan_sample(sm; 
                       data = texas_data,
                       num_samples = 1500,
                       num_warmups = 1500,
                       num_chains = 4,  
                       delta = 0.999, # Micheal Betancourt is going to be so mad at me for this
                       max_depth = 15,
                       seed = 47318)

if success(rc_texas)
  diagnose(sm)
  params_out = DataFrame(read_samples(sm))
  gq = stan_generate_quantities(sm)
end







#### Code chunk 4 ####
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

get_qs(y_pre_fit::Matrix{Float64}, y_est::Matrix{Float64}) = get_qs(DataFrame(y_pre_fit, :auto), DataFrame(y_est, :auto))



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



# Plot outcome. Start by getting matrices of draws from pre- and post-treatment period
texas_fit = gq[:, 1:length(y_texas_pre)]
texas_est = gq[:, (length(y_texas_pre) + 1):(length(y_texas_pre) + length(y_texas_post))]
texas_m, texas_ll, texas_ul = get_qs(texas_fit, texas_est)


texas_trend = plot_base("Synthetic control estimates: Mixtape example of\nblack prison population in Texas",
                        "Number of black prisoners (10,000s)",
                        "Year";
                        top_margin = 5mm)
plot!(texas_trend, Int.(unique(df_texas.year)), texas_m, 
      ribbon = (texas_m - texas_ll, texas_ul - texas_m),
      label = "Estimated trend",
      lw = 3,
      lc = :blue, fill = :blue)
plot!(texas_trend, Int.(unique(df_texas.year)), vcat(y_texas_pre, y_texas_post), 
      label = "Actual", 
      lc = :black,
      lw = 3);
vline!(texas_trend, [1993],
       lc = :blue, linestyle = :dash,
       linealpha = 0.5,
       label = "")
annotate!(texas_trend, 1992.9, 5.5, 
          StatsPlots.text("Policy change",
          10, :right))
annotate!(texas_trend, 2000, -0.8,
          StatsPlots.text("Source: https://raw.github.com/scunning1975/mixtape/master/texas.dta
          Analysis by sjwild.github.io",
          :right, 7))
png(texas_trend, "Images/Texas/texas_synthetic_control_estimates")











#### Code chunk 5 ####
# Download data and prep it for later steps
# Load data for Canada
# ensure variable names are the same across Canada and US data
url_canada = "https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv"
df_canada = CSV.read(HTTP.get(url_canada).body, DataFrame)
rename!(df_canada, :prname => :Province_State,
                   :numtoday => :cases,
                   :numconf => :total_cases)
df_canada = df_canada[ in([1, 99]).(df_canada.pruid) .== false, :] # Keep only the provincial data



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
# Data was download from websites of Statistics Canada and the US Census Bureau, then pre-cleaned to make it easier to use here
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


# Produce plot showing cases per day and then showing seven-day moving average
# First, combine cases data and population data
# Then filter
date_list = unique(df.date)[2:end]
df.newdate = [join(["X", df.date[i]]) for i in 1:size(df, 1)]
df.newdate = replace.(df.newdate, "-" => "_")
df = unstack(df[:, [:newdate, :Province_State, :total_cases]], :newdate, :total_cases)
df = innerjoin(df, pop, on = :Province_State)





# Get values for plots
pop_scale = 100_000     # to scale
y_raw = diff(Matrix{Float64}(df[df.Province_State .== "Alberta", 2:(end-1)]), dims = 2) ./ df[df.Province_State .== "Alberta", end] * pop_scale
X_raw =  diff(Matrix{Float64}(df[df.Province_State .!= "Alberta", 2:(end-1)]), dims = 2) ./ df[df.Province_State .!= "Alberta", end] * pop_scale





#### Code chunks for image. Code chunks do not appear in post ####
# For loop for all provinces and states except Alberta
pr = unique(df_stringency.RegionName)


# Base plot
ab_plot = plot(title = "Stringency index: Alberta vs. other provinces and states",
               title_position = :left,
               xrotation = 45,
               legend = nothing,
               grid = nothing,
               tickfontsize = 10,
               guidefontsize = 12,
               size = (750, 450),
               bottom_margin = 20mm,
               left_margin = 5mm)



for p in 1:length(pr)
    pr_name = pr[p]

    if pr_name != "Alberta"
        @df df_stringency[(df_stringency.RegionName .== pr_name) .&
             (start_date .≤ df_stringency.Date .≤ end_date), :] plot!(ab_plot, :Date, :StringencyIndex,
        lw = 1,
        lc = :grey,
        lalpha = .3)
    end

end

# Add Alberta and annotations
@df df_stringency[(df_stringency.RegionName .== "Alberta") .&
          (start_date .≤ df_stringency.Date .≤ end_date), :] plot!(ab_plot, :Date, :StringencyIndex,
      lw = 3,
      lc = :red,
      ylabel = "Index value")
annotate!(ab_plot, Date(2021, 09, 30), -8,
          StatsPlots.text("Source: Oxford COVID-19 Government Response Tracker,
          Blavatnik School of Government, University of Oxford
          Analysis by sjwild.github.io",
          :right, 7))
annotate!(ab_plot, Date(2021, 07, 05), 90,
          StatsPlots.text("Most guidelines dropped",
                          :left, 8))
vline!(ab_plot, [Date(2021, 07, 01)],
       lc = :red,
       linestyle = :dash,
       lalpha = 0.5)
png(ab_plot, "Images/Alberta/AB_stringency_index")   # save image





# Plot cases per day
cp_plot = plot(ylabel = "Cases per 100 thousand",
               title = "New cases per day in Alberta\nbefore and after July 1, 2021",
               title_position = :left,
               xrotation = 45,
               legend = nothing,
               grid = nothing,
               tickfontsize = 10,
               guidefontsize = 12,
               size = (750, 450),
               bottom_margin = 17mm,
               left_margin = 5mm,
               top_margin = 5mm)

for i in 1:size(X_raw, 1)
    plot!(cp_plot, date_list, X_raw[i, :],
          lc = :grey50,
          lalpha = 0.3)
end

plot!(cp_plot, date_list, y_raw[1, :],
      lc = :red,
      linewidth = 2)
annotate!(cp_plot, Date(2021, 07, 05), 375,
          StatsPlots.text("Most guidelines dropped",
                          :left, 8))
vline!(cp_plot, [Date(2021, 07, 01)],
       lc = :red,
       linestyle = :dash,
       lalpha = 0.5)
annotate!(cp_plot, Date(2021, 09, 30), -415,
      StatsPlots.text("Source: Government of Canada and CDC
      Analysis by sjwild.github.io",
      :right, 7))
png(cp_plot, "Images/Alberta/AB_cases_per_capita")





# Build matrices for 7-day moving average
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

# Plot
ra_plot = plot(ylabel = "Cases per 100 thousand",
               title = "7-day moving avergage of cases in Alberta,\nbefore and after July 1, 2021",
               title_position = :left,
               xrotation = 45,
               legend = nothing,
               grid = nothing,
               tickfontsize = 10,
               guidefontsize = 12,
               size = (750, 450),
               bottom_margin = 17mm,
               left_margin = 5mm,
               top_margin = 5mm)
for i in 1:63
  plot!(ra_plot, date_list[7:end], X[i, :], lc = :grey50, lalpha = 0.3)
end

plot!(ra_plot, date_list[7:end], y, 
      lc = :red, 
      linewidth = 2)
vline!(ra_plot, [Date(2021, 07, 01)],
       lc = :red,
       linestyle = :dash,
       lalpha = 0.5)
annotate!(ra_plot, Date(2021, 09, 30), -100,
          StatsPlots.text("Source: Government of Canada and CDC
          Analysis by sjwild.github.io",
          :right, 7))
annotate!(ra_plot, Date(2021, 07, 05), 150,
          StatsPlots.text("Most guidelines dropped",
                          :left, 8))
png(ra_plot, "Images/Alberta/AB_cases_7_day_average")





#### Code chunk 7 ####
#total number of cases
y_direct = Matrix{Float64}(df[df.Province_State .== "Alberta", 9:(end-1)]) ./ df[df.Province_State .== "Alberta", end] * pop_scale
X_direct = Matrix{Float64}(df[df.Province_State .!= "Alberta", 9:(end-1)]) ./ df[df.Province_State .!= "Alberta", end] * pop_scale


# get names of all provinces and states that have a higher stringency on July 1,
# which is the date AB "opened for summer"
filter!(row -> row.Date .== Date(2021, 07, 01), df_stringency)
ab = df_stringency.StringencyIndex[df_stringency.RegionName .== "Alberta"]

# get value of AB stringecy index on July 1, 2021
stringency_idx = df_stringency.RegionName[df_stringency.StringencyIndex .> ab]

# Remove states with zero reported cases
indx = [sum(X_direct[i, :] .≤ 0) for i in 1:63]
X_direct = X_direct[indx .== 0, :]

# Get names of provinces and states with stringency 
# index who are still in the dataset
Province_State = df.Province_State[df.Province_State .!= "Alberta"]
Province_State = Province_State[indx .== 0]

# Keep states and provinces with stringency index greater than AB
Province_State_idx = in(stringency_idx).(Province_State)


# Drop states with either excessive zeros or when stringency value is above thereshold
X_direct = X_direct[Province_State_idx, :]


# Log vales so we have a log-log model
y_direct = log.(y_direct)
X_direct = log.(X_direct)


# Fit normal model
# Build pre and post values
trt_value = 91
y_pre_direct = y_direct[1:trt_value]
y_post_direct = y_direct[(trt_value+1):end]
X_pre_direct = X_direct[:, 1:trt_value]
X_post_direct = X_direct[:, (trt_value+1):end]
y_mean_direct = y_pre_direct[end]
x_mean_direct = X_pre_direct[:, end]


# Remove means to centre values
y_pre_direct = (y_pre_direct .- y_mean_direct) 
X_pre_direct = (X_pre_direct .- x_mean_direct) 
y_post_direct = (y_post_direct .- y_mean_direct) 
X_post_direct = (X_post_direct .- x_mean_direct) 

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
                           max_depth = 20, # for some reason this was needed on an earlier version. Something is clearly wrong.
                           seed = 33294)


# check diagnostics and extract parameters
if success(rc_AB_direct)
    diagnose(sm)
    params_out = DataFrame(read_samples(sm))
    gq = stan_generate_quantities(sm)
end

y_fit_direct = gq[:, 1:length(y_pre_direct)] .+ y_mean_direct 
y_post_est_direct = gq[:, (length(y_pre_direct) + 1):(length(y_pre_direct) + length(y_post_direct))] .+ y_mean_direct 
y_fit_direct = exp.(y_fit_direct)
y_post_est_direct = exp.(y_post_est_direct)
AB_m_direct, AB_ll_direct, AB_ul_direct = get_qs(y_fit_direct, y_post_est_direct)

y_actual = Matrix{Float64}(df[df.Province_State .== "Alberta", 9:(end-1)]) ./ df[df.Province_State .== "Alberta", end] * pop_scale
y_actual = Vector(y_actual[1, :])

date_list = [(start_date + Day(7)):Day(1):end_date;]

# Plot directly modeled cumulative cases
AB_trend_direct = plot_base("Acutal vs estimated cases: Directly modelling\ncumulative cases",
                            "Num cases (per 100k)",
                            "Date")
plot!(AB_trend_direct, date_list, AB_m_direct, 
      ribbon = (AB_m_direct - AB_ll_direct, AB_ul_direct - AB_m_direct),
      lc = :red, fill = :red, lw = 3,
      label = "Estimated trend";
      top_margin = 5mm)
plot!(AB_trend_direct, date_list, y_actual, 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend_direct, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
hline!(AB_trend_direct, [y_actual[91]],
       lc = :black, linestyle = :dash,
       linealpha = 0.5,
       lw = 2,
       label = "")
annotate!(AB_trend_direct, Date(2021, 07, 03), 8000, 
          StatsPlots.text("Most guidelines dropped",
          10, :left))
annotate!(AB_trend_direct, Date(2021, 09, 30), 750,
          StatsPlots.text("Source: Government of Canada and CDC
          Analysis by sjwild.github.io",
          :right, 7))
png(AB_trend_direct, "Images/Alberta/directly_modelling_cumulative_cases")







#### Code chunk 8 ####
# Fit normal model
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


# Build indices to remove zero cases and states/provinces with lower stringency
indx = [sum(X[i, :] .≤ 0) for i in 1:63]
X = X[indx .== 0, :]
Province_State = df.Province_State[df.Province_State .!= "Alberta"]
Province_State = Province_State[indx .== 0]

# Keep states and provinces with stringency index greater than AB
Province_State_idx = in(stringency_idx).(Province_State)


# Subset X and build pre and post matrices
X = X[Province_State_idx, :]
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




#### Code chunk 9 ####
# Transform parameters
y_fit = gq[:, 1:length(y_pre)]
y_post_est = gq[:, (length(y_pre) + 1):(length(y_pre) + length(y_post))]
y_fit = exp.(y_fit)
y_post_est = exp.(y_post_est)
  

# Build series for plotting.
# Start with pre-period fitted values
AB_m, AB_ll, AB_ul = get_qs(y_fit, y_post_est)

# dates
AB_trend = plot_base("Actual vs estimated cases using Alberta\n7-day moving average",
                     "Num cases (per 100k)",
                     "Date";
                     legend = :bottomleft,
                     topmargin = 5mm)
plot!(date_list, AB_m, 
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
          StatsPlots.text("Most guidelines dropped",
          10, :left))
annotate!(AB_trend, Date(2021, 09, 30), -25,
          StatsPlots.text("Source: Government of Canada and CDC
          Analysis by sjwild.github.io",
          :right, 7))
png(AB_trend, "Images/Alberta/estimates_7_day_moving_average")





#### Code chunk 10 ####
# Build counts by adding in values
cases_start = 3400.10

# Transform parameters and add
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
AB_m_c, AB_ll_c, AB_ul_c = get_qs(y_fit, y_post_est)


AB_trend_c = plot_base("Cumulative cases based on 7-day moving average",
                       "Num cases (per 100k)",
                       "Date";
                       legend = :topleft,
                       topmargin = 5mm)
plot!(AB_trend_c, date_list, AB_m_c, 
      ribbon = (AB_m_c - AB_ll_c, AB_ul_c - AB_m_c),
      lc = :red, fill = :red, lw = 3,
      label = "Estimated trend")
plot!(AB_trend_c, date_list, y_actual, 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend_c, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
annotate!(AB_trend_c, Date(2021, 07, 03), 6500, 
          StatsPlots.text("Most guidelines dropped",
          10, :left))
annotate!(AB_trend_c, Date(2021, 09, 30), 1750,
          StatsPlots.text("Source: Government of Canada and CDC
          Analysis by sjwild.github.io",
          :right, 7))
png(AB_trend_c, "Images/Alberta/cumulative_cases_using_7_day_moving_average")




#### Code chunk 11 ####
# Define arcsinh function
ihs(x) = log(x + sqrt(x^2 + 1))


# Subset data for modelling
X_raw = X_raw[indx .== 0, 7:end]
X_raw = X_raw[Province_State_idx, :]
y_raw = y_raw[7:end]


# Build pre and post values
y_pre_raw = y_raw[1:trt_value] 
y_post_raw = y_raw[(trt_value+1):end] 
X_pre_raw = X_raw[:, 1:trt_value]
X_post_raw = X_raw[:, (trt_value+1):end]

AB_data_ihs = Dict(
  
  "N_pre" => size(X_pre_raw, 2), # Numbers of pre observations
  "N_post" => size(X_post_raw, 2), # Number of post observations
  "p" => size(X_pre_raw, 1), # Size of donor pool
  "y_pre" => ihs.(y_pre_raw),
  "X_pre" => ihs.(X_pre_raw),
  "X_post" => ihs.(X_post_raw),
  
  "scale_alpha" => 1,
  "scale_g" => 1,
  
  "slab_scale" => 1,
  "slab_df" => 5,
  
  "nu_g" => 3,
  "nu_l" => 3
  
)

# run model
rc_ihs = stan_sample(sm;
                     data = AB_data_ihs,
                     num_samples = 1500,
                     num_warmups = 1500,
                     num_threads = 4, 
                     delta = 0.999,
                     max_depth = 15,
                     seed = 21643)


if success(rc_ihs)
    diagnose(sm)
    params_out_ihs = DataFrame(read_samples(sm))
    gq_ihs = stan_generate_quantities(sm)
end

y_fit_ihs = exp.(gq_ihs[:, 1:length(y_pre_raw)]) ./ 2
y_post_est_ihs = exp.(gq_ihs[:, (length(y_pre_raw) + 1):(length(y_pre_raw) + length(y_post_raw))]) ./ 2
AB_m_ihs, AB_ll_ihs, AB_ul_ihs = get_qs(y_fit_ihs, y_post_est_ihs)


# Plot directly modeled cumulative cases
AB_trend_ihs = plot_base("Actual vs estimated daily COVID cases in Alberta",
                        "Num cases (per 100k)",
                        "Date";
                        ylims = (0, 90))
plot!(AB_trend_ihs, date_list, AB_m_ihs, 
    ribbon = (AB_m_ihs - AB_ll_ihs, AB_ul_ihs - AB_m_ihs),
    lc = :red, fill = :red, lw = 3,
    label = "Estimated trend";
    top_margin = 5mm)
plot!(AB_trend_ihs, date_list, y_raw, 
    label = "Actual", 
    lc = :black,
    lw = 3)
vline!(AB_trend_ihs, [Date(2021, 07, 01)],
    lc = :red, linestyle = :dash,
    linealpha = 0.5,
    label = "")
annotate!(AB_trend_ihs, Date(2021, 07, 03), 80, 
        StatsPlots.text("Most guidelines dropped",
        10, :left))
annotate!(AB_trend_ihs, Date(2021, 09, 30), -38,
          StatsPlots.text("Source: Government of Canada and CDC
          Analysis by sjwild.github.io",
          :right, 7))
png(AB_trend_ihs, "Images/Alberta/daily_cases_ihs")


# Now add in cases to get cumulative cases
y_fit_ihs[:, 1] = y_fit_ihs[:, 1] .+ cases_start
for m in 2:size(y_fit_ihs, 2)
    for n in 1:size(y_fit_ihs, 1)
      y_fit_ihs[n, m] = sum(y_fit_ihs[n, (m-1):m])
    end
end
  
y_post_est_ihs[:, 1] = y_fit_ihs[:, end] + y_post_est_ihs[:, 1]
  
for m in 2:size(y_post_est_ihs, 2)
    for n in 1:size(y_post_est_ihs, 1)
      y_post_est_ihs[n, m] = sum(y_post_est_ihs[n, (m-1):m])
    end
end



AB_m_ihs_c, AB_ll_ihs_c, AB_ul_ihs_c = get_qs(y_fit_ihs, y_post_est_ihs)


AB_trend_ihs_c = plot_base("Estimated vs actual cumulative cases in Alberta",
                           "Cumulative cases (per 100k)",
                           "Date";
                           legend = :topleft)
plot!(AB_trend_ihs_c, date_list, AB_m_ihs_c, 
        ribbon = (AB_m_ihs_c - AB_ll_ihs_c, AB_ul_ihs_c - AB_m_ihs_c),
        lc = :red, fill = :red, lw = 3,
        label = "Estimated trend")
plot!(AB_trend_ihs_c, date_list, y_actual, 
      label = "Actual", 
      lc = :black,
      lw = 3)
vline!(AB_trend_ihs_c, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
annotate!(AB_trend_ihs_c, Date(2021, 07, 03), 6800, 
            StatsPlots.text("Most guidelines dropped",
            10, :left))
annotate!(AB_trend_ihs_c, Date(2021, 09, 30), 1500,
            StatsPlots.text("Source: Government of Canada and CDC
            Analysis by sjwild.github.io",
            :right, 7))
png(AB_trend_ihs_c, "Images/Alberta/cumulative_cases_by_daily_cases")






#### Code chunk 12 ####
# Get difference in cumulative cases
y_cumulative = [y_fit_ihs y_post_est_ihs]
N, M = size(y_cumulative)
y_difference = Matrix{Float64}(undef, N, M)
for n in 1:N
    for m in 1:M
        y_difference[n, m] = y_actual[m] - y_cumulative[n, m]
    end
end


AB_m_d, AB_ll_d, AB_ul_d = get_qs(y_difference[:, 1:91], y_difference[:, 92:end])


AB_trend_d = plot_base("Difference in cases in Alberta",
                     "Difference (per 100k)",
                     "Date";
                     legend = :topleft)
plot!(AB_trend_d, date_list, AB_m_d, 
        ribbon = (AB_m_d - AB_ll_d, AB_ul_d - AB_m_d),
        lc = :red, fill = :red, lw = 3,
        label = "Estimated trend")
vline!(AB_trend_d, [Date(2021, 07, 01)],
       lc = :red, linestyle = :dash,
       linealpha = 0.5,
       label = "")
hline!(AB_trend_d, [0],
       lc = :black, linestyle = :dash,
       linealpha = 0.5,
       lw = 2,
       label = "")
annotate!(AB_trend_d, Date(2021, 07, 03), 600, 
            StatsPlots.text("Most guidelines dropped",
            10, :left))
annotate!(AB_trend_d, Date(2021, 09, 30), -1150,
            StatsPlots.text("Source: Government of Canada and CDC
            Analysis by sjwild.github.io",
            :right, 7))
png(AB_trend_d, "Images/Alberta/difference_in_cumulative_cases")
