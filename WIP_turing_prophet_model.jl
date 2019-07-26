using Turing, StatsPlots, Random, CSV, DataFrames

Random.seed!(12)

# https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/

nchangepoints = 10
t = collect(1:1000)
changepoints = sort(randperm(1000)[1:nchangepoints])

# Create the matrix A
# A is a t X nchangepoints matrix
A = [(t[i] >= changepoints[j])*1 for i=t, j=1:nchangepoints]

delta = rand(Normal(0, 1), nchangepoints)
k = 1
m = 5

growth = (k .+ A*delta).*t
gamma = -changepoints.*delta
offset = m .+ A * gamma
trend = growth + offset


plot(trend)
plot(t, growth)
plot(t, offset)

#
df = CSV.read("data/example_wp_log_peyton_manning.csv")

df[!, :yscale] = df[!, :y] ./ maximum(df[!, :y])


df[!, :t] = (df.ds - minimum(df.ds)) / (maximum(df.ds) - minimum(df.ds))

plot(df.ds, df.yscale)

# Proportion of history in which trend changepoints will be estimated.
changepoints_range = .8
# The standard deviation of the prior on the growth.
growth_prior_scale = 5
# The scale of the Laplace prior on the delta vector.
changepoints_prior_scale = .05

# Declare our Turing model.
@model trendmodel(t, y) = begin

    # num observations
    nobs = length(t)
    s = 1:nchangepoints+1:(changepoints_range * maximum(t))
    A = [(t[i] >= s[j])*1 for i=1:nobs, j=1:nchangepoints]

    k ~ Normal(0, growth_prior_scale)

    # rate of change
    delta = Array{Real}(undef, nchangepoints)
    delta ~ [Laplace(0, changepoints_prior_scale)]

    # offset
    m ~ Normal(0, 5)

    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    gamma = -s.*delta

    g = (k .+ A*delta).*t + m .+ A * gamma

    # The number of observations.
    sd = absolute(Cauchy(0, .5))
    for n in 1:nobs
        y[n] ~ Normal(g[n], sd)
    end
end

model = trendmodel(df.t, df.yscale)

chain = sample(model, NUTS(1500, 200, 0.65));

# def trend_model(m, t, n_changepoints=25, changepoints_prior_scale=0.05,
#                 growth_prior_scale=5, changepoint_range=0.8):
#     """
#     The piecewise linear trend with changepoint implementation in PyMC3.
#     :param m: (pm.Model)
#     :param t: (np.array) MinMax scaled time.
#     :param n_changepoints: (int) The number of changepoints to model.
#     :param changepoint_prior_scale: (flt/ None) The scale of the Laplace prior on the delta vector.
#                                     If None, a hierarchical prior is set.
#     :param growth_prior_scale: (flt) The standard deviation of the prior on the growth.
#     :param changepoint_range: (flt) Proportion of history in which trend changepoints will be estimated.
#     :return g, A, s: (tt.vector, np.array, tt.vector)
#     """
#     s = np.linspace(0, changepoint_range * np.max(t), n_changepoints + 1)[1:]
#
#     # * 1 casts the boolean to integers
#     A = (t[:, None] > s) * 1
#
#     with m:
#         # initial growth
#         k = pm.Normal('k', 0 , growth_prior_scale)
#
#         if changepoints_prior_scale is None:
#             changepoints_prior_scale = pm.Exponential('tau', 1.5)
#
#         # rate of change
#         delta = pm.Laplace('delta', 0, changepoints_prior_scale, shape=n_changepoints)
#         # offset
#         m = pm.Normal('m', 0, 5)
#         gamma = -s * delta
#
#         g = (k + det_dot(A, delta)) * t + (m + det_dot(A, gamma))
#     return g, A, s
