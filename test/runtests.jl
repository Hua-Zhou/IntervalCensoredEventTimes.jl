module PkgTest
using IntervalCensoredEventTimes
using CSV, DataFrames, Ipopt, NLopt, SparseArrays, Random, Test, UnicodePlots

@testset "Breast Cancer" begin
bcfile = joinpath(@__DIR__, "..", "data", "breast_cancer.csv")
df  = CSV.File(bcfile) |> DataFrame
g   = Float64.(df[!, :Group] .== "RCT")
L   = Float64.(df[!, :L])
R   = map(x -> ismissing(x) ? Inf : Float64(x), df[!, :R])
icm = IntervalCensoredModel(reshape(g, length(g), 1), L, R)
display([g icm.L icm.R icm.C]); println()
@info "NPMLE of survival functin S(t)"
npmle!(icm)
plt = lineplot([0; icm.ts], [1; icm.S₀], xlabel = "Time by Months", ylabel = "Survival Functions")
display(plt); println()
@info "Log-likelihood at starting point"
initialize_uniform!(icm)
display(DataFrame(t=icm.ts, St=icm.S₀, Λ₀t=icm.Λ₀, λ₀t=icm.λ₀)); println()
println("logl = $(loglikelihood!(icm))")
@info "MLE of PH model"
# solver = NLopt.NLoptSolver(algorithm = :LN_BOBYQA, ftol_rel = 1e-12, ftol_abs = 1e-8, maxeval = 10000)
solver = Ipopt.IpoptSolver(print_level=5)
# solver = NLopt.NLoptSolver(algorithm = :LN_BOBYQA)
# solver = NLopt.NLoptSolver(algorithm = :LN_COBYLA, maxeval = 10000)
# solver = NLopt.NLoptSolver(algorithm = :LD_MMA)
# solver = NLopt.NLoptSolver(algorithm = :LD_SLSQP)
# solver = NLopt.NLoptSolver(algorithm = :LD_LBFGS)
fit!(icm, solver, init = initialize_uniform!(icm))
display(icm)
@info "IHT"
Random.seed!(123)
X = randn(length(icm.C), 100)
k = 10
icsm = IntervalCensoredSparseModel(icm, X, k)
iht!(icsm, maxiter=5, verbose=true, debiasing=true, printfreq=1)
βx = sparsevec(icsm.βx)
display(DataFrame(j=βx.nzind, βj=βx.nzval)); println()
end

end
