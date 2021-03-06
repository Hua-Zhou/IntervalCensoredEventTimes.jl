module IntervalCensoredEventTimes

using LinearAlgebra, SparseArrays
using Distributions, Ipopt, MathProgBase, StatsModels, UnicodePlots
import LinearAlgebra: BlasReal, copytri!, mul!

export fit!, iht!, initialize_uniform!, IcmAlphaMatrix, 
IntervalCensoredModel, IntervalCensoredSparseModel,
loglikelihood!, npmle!

"""
IntervalCensoredModel

Interval censored event time model, which contains data, model parameters, 
and working arrays.

IntervalCensoredModel(Z, L, R)

# Positional arguments  
- `Z`: `n`-by-`p` covariate matrix, excluding intercept.  
- `L`: `n` vector of left censoring times.
- `R`: `n` vector of right censoring times.

# Keyword arguments
"""
struct IntervalCensoredModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # Data
    Z          :: Matrix{T} # n-by-p covariate matrix
    L          :: Vector{T} # n vector of left ends of interval
    R          :: Vector{T} # n vector of right ends of interval
    C          :: Vector{Symbol}    # n vector of censoring type
#    A          :: IcmAlphaMatrix{T} # αij=I(Li<sj≤Ri) indicator matrix
    # parameters
    β          :: Vector{T}   # regression coefficients    
    Λ₀         :: Vector{T}   # baseline cumulative hazards
    λ₀         :: Vector{T}   # baseline hazards
    S₀         :: Vector{T}   # baseline survival function
    # working arrays
    ts         :: Vector{T}   # finite observed time points in the data
    Lidx       :: Vector{Int} # ts[Lidx] == L
    Ridx       :: Vector{Int} # ts[Ridx] == R
    η          :: Vector{T}   # η = Z * β
    expη       :: Vector{T}   # expη = exp(η)
    # gradient
    ∇λ₀        :: Vector{T}
    ∇β         :: Vector{T}
    βres       :: Vector{T}
    # Hessian
    Hλ₀λ₀      :: Matrix{T}
    Hββ        :: Matrix{T}
    Hλ₀β       :: Matrix{T}
    Vββ        :: Matrix{T}
    glmwt      :: Vector{T}
    # logical
    isfitted   :: Vector{Bool}
    # scrach spaces
    storage_n  :: Vector{T}
    storage_np :: Matrix{T}
end

function IntervalCensoredModel(
    Z :: AbstractMatrix{T}, 
    L :: Vector{T}, 
    R :: Vector{T}
    ) where T <: BlasReal
    n, p = size(Z)
    @assert length(L) == n "length(L) ≠ size(Z, 1); check input"
    @assert length(R) == n "length(R) ≠ size(Z, 1); check input"
    @assert all(L .≤ R) "Some L[i] are larger than R[i]; check input"
    @assert all(L .≥ 0) "Some L[i] are negative; check input"
    @assert all(R .≥ 0) "Some R[i] are negative; check input"
    ts   = unique!(sort!([L; R])) # last time can be infinity
    iszero(ts[1])  && popfirst!(ts)
    isinf(ts[end]) && pop!(ts)
    m = length(ts)
    # Lidx[i] = 0 if L[i] = 0
    Lidx = map(x -> isnothing(x) ? 0 : x, indexin(L, ts)) 
    # Ridx[i] = length(ts) + 1 if R[i] = Inf
    Ridx = map(x -> isnothing(x) ? length(ts) + 1 : x, indexin(R, ts))
    C    = Vector{Symbol}(undef, n)
    for i in 1:n
        if iszero(L[i]) # L[i] == 0
            if isfinite(R[i])
                C[i] = :left_censored
            else
                C[i] = :non_informative
            end
        elseif isfinite(L[i]) # 0 < L[i] < ∞
            if R[i] == L[i]
                C[i] = :exact_time
            elseif isfinite(R[i])
                C[i] = :interval_censored
            else
                C[i] = :right_censored
            end
        else # L[i] = ∞
            C[i] = :non_informative
        end
    end
    η        = Vector{T}(undef, n)
    expη     = Vector{T}(undef, n)
    β        = Vector{T}(undef, p)
    Λ₀       = Vector{T}(undef, m)
    λ₀       = Vector{T}(undef, m)
    S₀       = Vector{T}(undef, m)
    ∇λ₀      = Vector{T}(undef, m)
    ∇β       = Vector{T}(undef, p)
    βres     = Vector{T}(undef, n)
    Hλ₀λ₀    = Matrix{T}(undef, m, m)
    Hββ      = Matrix{T}(undef, p, p)
    Hλ₀β     = Matrix{T}(undef, m, p)
    Vββ      = Matrix{T}(undef, p, p)
    glmwt    = Vector{T}(undef, n)
    isfitted = [false]
    storage_n  = Vector{T}(undef, n)
    storage_np = Matrix{T}(undef, n, p)
    # constructor
    IntervalCensoredModel{T}(
        Z, L, R, C,
        β, Λ₀, λ₀, S₀, ts, Lidx, Ridx, η, expη,
        ∇λ₀, ∇β, βres, 
        Hλ₀λ₀, Hββ, Hλ₀β, Vββ, glmwt, isfitted,
        storage_n, storage_np)
end

coefnames(icm::IntervalCensoredModel) = "Z" .* string.(1:size(icm.Z, 2))
coef(icm::IntervalCensoredModel) = icm.β
nobs(icm::IntervalCensoredModel) = length(icm.L)
stderror(icm::IntervalCensoredModel) = [sqrt(icm.Vββ[i, i]) for i in 1:size(icm.Z, 2)]
vcov(icm::IntervalCensoredModel) = icm.Vββ

confint(icm::IntervalCensoredModel, level::Real) = hcat(coef(icm), coef(icm)) +
    stderror(icm) * quantile(Normal(), (1. - level) / 2.) * [1. -1.]
confint(icm::IntervalCensoredModel) = confint(icm, 0.95)

function coeftable(icm::IntervalCensoredModel)
    mstder = stderror(icm)
    mcoefs = coef(icm)
    wald = mcoefs ./ mstder
    pvals = 2 * Distributions.ccdf.(Normal(), abs.(wald))
    StatsModels.CoefTable(hcat(mcoefs, mstder, wald, pvals),
        ["Estimate", "Std. Error", "Z", "Pr(>|Z|)"],
        coefnames(icm), 4, 3)
end

function Base.show(io::IO, icm::IntervalCensoredModel)
    println(io)
    println(io, "Interval Censored Event Time PH Model")
    println(io)
    println(io, "Total observations: $(length(icm.L))")
    println(io, "Left-censored     : $(count(x -> x == :left_censored, icm.C))")
    println(io, "Right-censored    : $(count(x -> x == :right_censored, icm.C))")
    println(io, "Interval-censored : $(count(x -> x == :interval_censored, icm.C))")
    println(io, "Exact time        : $(count(x -> x == :exact_time, icm.C))")
    println(io)
    if icm.isfitted[1]
        println(io, "Log-likelihood: $(loglikelihood!(icm))")
        println(io)
        println(io, "Regression coefficients:")
        show(io, coeftable(icm))
        println(io)
        println(io)
        println(io, "Baseline survival:")
        show(io, lineplot([0; icm.ts], [1; icm.S₀], 
            xlabel = "Time", ylabel = "Survival Functions"))
    else
        println("The regression model has not been fit.")
        return nothing
    end
    println(io)
    println(io)
    nothing
end

"""
    ICM_Alpha_Matrix(icm::IntervalCensoredModel)

An indicator matrix constructed from an interval censored model. 
- If `C[i]=:left-censored`, the correspondng row is 1{sj≤Ri}.  
- If `C[i]=:right-censored`, the correspondng row is 1{sj≤Li}.  
- If `C[i]=:interval-censored`, it generates **two rows**: 1{sj≤Li} and 1{Li<sj≤Ri}.  
- If `C[i]=:exact_time`, the correspondng row is 1{sj≤Li}.
"""
struct IcmAlphaMatrix{T <: Real} <: AbstractMatrix{T}
    C     :: Vector{Symbol}
    s     :: Vector{T} # examination time points: 0 < s1 < ... < sm < ∞
    Lidx  :: Vector{Int}
    Ridx  :: Vector{Int}
end

function IcmAlphaMatrix(icm::IntervalCensoredModel{T}) where T <: Real
    # constructor
    IcmAlphaMatrix{T}(icm.C, icm.ts, icm.Lidx, icm.Ridx)
end

LinearAlgebra.issymmetric(::IcmAlphaMatrix) = false
Base.size(A::IcmAlphaMatrix) = (
    length(A.C) + count(ct -> ct == :interval_censored, A.C), length(A.ts)
    )

function LinearAlgebra.mul!(
    out :: Vector{T}, 
    A   :: IcmAlphaMatrix{T}, 
    v   :: Vector{T}
    ) where T <: Real
    fill!(out, 0)
    row = 1
    @inbounds for i in 1:length(A.C)
        if A.C[i] == :left_censored
            for j in 1:A.Ridx[i]
                out[row] += v[j]
            end
            row += 1
        elseif A.C[i] == :right_censored
            for j in 1:A.Lidx[i]
                out[row] += v[j]
            end
            row += 1
        elseif A.C[i] == :interval_censored
            for j in 1:A.Lidx[i]
                out[row] += v[j]
            end
            row += 1
            for j in (A.Lidx[i]+1):A.Ridx[j]
                out[row] += v[j]
            end
            row += 1
        elseif A.C[i] == :exact_time
            for j in 1:A.Lidx[i]
                out[row] += v[j]
            end
            row += 1            
        end
    end
    out
end

function LinearAlgebra.mul!(
    out :: Vector{T}, 
    At  :: Transpose{T, IcmAlphaMatrix{T}},
    v   :: Vector{T}
    ) where T <: Real
    A = At.parent
    fill!(out, 0)
    row = 0
    for i in 1:A.nrows
        vrow = v[row]
        if A.C[i] == :left_censored
            for j in 1:A.Ridx[i]
                out[j] += vrow
            end
            row += 1
        elseif A.C[i] == :right_censored
            for j in 1:A.Lidx[i]
                out[j] += vrow
            end
            row += 1
        elseif A.C[i] == :interval_censored
            for j in 1:A.Lidx[i]
                out[j] += vrow
            end
            row += 1
            for j in (A.Lidx[i]+1):A.Ridx[j]
                out[j] += vrow
            end
            row += 1
        elseif A.C[i] == :exact_time
            for j in 1:A.Lidx[i]
                out[j] += vrow
            end
            row += 1            
        end
    end
    out
end

include("nlp.jl")
include("sparse.jl")

end # module
