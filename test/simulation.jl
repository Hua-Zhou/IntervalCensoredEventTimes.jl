module Simulation

using CSV, DataFrames, Distributions, IntervalCensoredEventTimes, 
Ipopt, NLopt, Random, SparseArrays, Statistics, Test

Random.seed!(123)
# number of observations
n = 1_000
# 2 covariates in base model: one binary and one continuous
Z  = [rand(Bernoulli(0.5), n) randn(n)]
βz = [1.0, 0.8]
# auto-correlated covariates for variable selection
p  = 100
ρ  = 0.1
Σx = [ρ^(abs(i - j)) for i in 1:p, j in 1:p]
X  = Matrix(transpose(rand!(MvNormal(Σx), Matrix{Float64}(undef, p, n))))
# first 5 coefficients are signal
βx = zeros(p)
βx[1:5] .= 0.25 
# systematic component
η  = Z * βz + X * βx
expη = exp.(-η)
# actual event times
T = rand.(Exponential.(expη))
# examination time points
s = quantile(T, 0.0:0.1:1.0)
s = [s[1]/2; s; s[end]+1]
display(s); println()
# censoring
C = rand([:left_censored, :right_censored, :interval_censored], n)
L = Vector{Float64}(undef, n)
R = Vector{Float64}(undef, n)
for i in 1:n
    if C[i] == :left_censored
        L[i] = 0
        R[i] = s[findfirst(sj -> sj ≥ T[i], s)]
    elseif C[i] == :right_censored
        L[i] = s[findlast(sj -> sj < T[i], s)]
        R[i] = Inf
    elseif C[i] == :interval_censored
        L[i] = s[findlast(sj -> sj < T[i], s)]
        R[i] = s[findfirst(sj -> sj ≥ T[i], s)]
    end
end
@info "Simulated interval-censored data:"
show(DataFrame(T=T, C=C, L=L, R=R)); println()
@info "Base model:"
icm = IntervalCensoredModel(Z, L, R)
solver = Ipopt.IpoptSolver(print_level=3)
@time fit!(icm, solver, init=initialize_uniform!(icm))
display(icm); println()
@info "Sparse model by IHT:"
icsm = IntervalCensoredSparseModel(icm, X, 5)
@time iht!(icsm, verbose=true, debiasing=true, printfreq=1)
βx = sparsevec(icsm.βx)
display(DataFrame(j=βx.nzind, βj=βx.nzval)); println()
end
