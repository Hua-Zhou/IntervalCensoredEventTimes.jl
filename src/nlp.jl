"""
    loglikelihood!(m::IntervalCensoredModel)

This function uses the parameter values `m.η`, `m.expη`, `m.λ₀`, `m.Λ₀`.
"""
function loglikelihood!(
    icm      :: IntervalCensoredModel{T},
    needgrad :: Bool = false,
    needhess :: Bool = false
    ) where T <: BlasReal
    n, p = size(icm.Z)
    # log-likelihood
    logl = zero(T)
    needgrad && fill!(icm.∇λ₀, 0)
    if needhess; fill!(icm.Hλ₀λ₀, 0); fill!(icm.Hλ₀β, 0); end
    @inbounds for i in 1:n
        if icm.C[i] == :left_censored
            s = exp(- icm.Λ₀[icm.Ridx[i]] * icm.expη[i])
            logl += log1p(-s)
            if needgrad
                g = icm.expη[i] / (inv(s) - 1)
                for j in 1:icm.Ridx[i]
                    icm.∇λ₀[j] += g
                end
                icm.βres[i] = icm.Λ₀[icm.Ridx[i]] * g
            end
            if needhess
                # Hλ₀λ₀
                h = abs2(icm.expη[i]) / (2 - s - inv(s))
                for j in 1:icm.Ridx[i], k in 1:j
                    icm.Hλ₀λ₀[k, j] -= h
                end
                # Hββ
                g = icm.Λ₀[icm.Ridx[i]] * icm.expη[i] / (inv(s) - 1)
                icm.glmwt[i] = - g * (1 - g * inv(s))
                # Hλ₀β
                g = icm.expη[i] / (inv(s) - 1) + icm.Λ₀[icm.Ridx[i]] * h
                for k in 1:p, j in 1:icm.Ridx[i]
                    icm.Hλ₀β[j, k] -= g * icm.Z[i, k]
                end
            end
        elseif icm.C[i] == :right_censored
            l     = - icm.Λ₀[icm.Lidx[i]] * icm.expη[i]
            logl += l 
            if needgrad
                for j in 1:icm.Lidx[i]
                    icm.∇λ₀[j] -= icm.expη[i]
                end
                icm.βres[i] = l
            end
            if needhess
                icm.glmwt[i] = -l
                for k in 1:p, j in 1:icm.Lidx[i]
                    icm.Hλ₀β[j, k] += icm.expη[i] * icm.Z[i, k]
                end
            end
        elseif icm.C[i] == :interval_censored
            ΔΛ₀  = icm.Λ₀[icm.Ridx[i]] - icm.Λ₀[icm.Lidx[i]]
            l     = - icm.Λ₀[icm.Lidx[i]] * icm.expη[i]
            s     = exp(- ΔΛ₀ * icm.expη[i])
            logl += l + log1p(-s)
            if needgrad
                g = icm.expη[i]
                for j in 1:icm.Lidx[i]
                    icm.∇λ₀[j] -= g
                end
                g = icm.expη[i] / (inv(s) - 1)
                for j in (icm.Lidx[i] + 1):icm.Ridx[i]
                    icm.∇λ₀[j] += g
                end
                icm.βres[i] = l + ΔΛ₀ * g
            end
            if needhess
                # Hλ₀λ₀
                h = abs2(icm.expη[i]) / (2 - s - inv(s))
                for j in (icm.Lidx[i]+1):icm.Ridx[i], k in (icm.Lidx[i]+1):j
                    icm.Hλ₀λ₀[k, j] -= h
                end
                # Hββ
                g = ΔΛ₀ * icm.expη[i] / (inv(s) - 1)
                icm.glmwt[i] = icm.Λ₀[icm.Lidx[i]] * icm.expη[i] - g * (1 - g * inv(s))
                # Hλ₀β
                for k in 1:p
                    for j in 1:icm.Lidx[i]
                        icm.Hλ₀β[j, k] += icm.expη[i] * icm.Z[i, k]
                    end
                    g = - icm.expη[i] / (inv(s) - 1) - ΔΛ₀ * h
                    for j in (icm.Lidx[i] + 1):icm.Ridx[i]
                        icm.Hλ₀β[j, k] += g * icm.Z[i, k]
                    end
                end
            end
        elseif icm.C[i] == :exact_time
            Tidx  = icm.Lidx[i]
            logl += log(icm.λ₀[Tidx]) + icm.η[i] - icm.Λ₀[Tidx] * icm.expη[i]
            if needgrad
                g = icm.expη[i]
                for j in 1:Tidx
                    icm.∇λ₀[j] -= g
                end
                icm.∇λ₀[Tidx] += inv(icm.λ₀[Tidx])
                icm.βres[i] = 1 - icm.Λ₀[Tidx] * icm.expη[i]
            end
            if needhess
                # Hλ₀λ₀
                icm.Hλ₀λ₀[Tidx, Tidx] += abs2(inv(icm.λ₀[Tidx]))
                # Hββ
                icm.glmwt[i] = icm.Λ₀[Tidx] * icm.expη[i]
                # Hλ₀β
                for k in 1:p, j in 1:Tidx
                    icm.Hλ₀β[j, k] += icm.expη[i] * icm.Z[i, k]
                end
            end
        end
    end
    needgrad && mul!(icm.∇β, transpose(icm.Z), icm.βres)
    if needhess
        copytri!(icm.Hλ₀λ₀, 'U')
        for (i, w) in enumerate(icm.glmwt)
            icm.glmwt[i] = w > 0 ? sqrt(w) : zero(w)
        end
        mul!(icm.storage_np, Diagonal(icm.glmwt), icm.Z)
        mul!(icm.Hββ, transpose(icm.storage_np), icm.storage_np)
    end
    # return log-likelihood
    logl
end

"""
    initialize_uniform!(icm::IntervalCensoredModel)

Set `icm.λ₀`, `icm.Λ₀` and `icm.S₀` according to a discrete uniform distribution 
on the observed examination time points `0 < s1 < ... < sm < ∞`. `icm.β` are 
set to  zero; `icm.η` and `icm.expη` are set accordingly.
"""
function initialize_uniform!(icm::IntervalCensoredModel{T}) where T <: Real
    m    = length(icm.ts)
    invm = inv(m)
    # survival function
    icm.S₀[end] = 0
    @inbounds for i in m-1:-1:1
        icm.S₀[i] = icm.S₀[i+1] + invm
    end
    # hazard function 
    icm.λ₀[1] = invm
    @inbounds for i in 2:m
        icm.λ₀[i] = invm / icm.S₀[i - 1]
    end
    # cumulative hazard
    cumsum!(icm.Λ₀, icm.λ₀)    
    # β, η, expη
    fill!(icm.β, 0)
    fill!(icm.η, 0)
    fill!(icm.expη, 1)
    icm
end

"""
    npmle!(icm::IntervalCensoredModel)

Non-parametric MLE (NPMLE) of survival function `S(t)` using the MM algorithm. 
`icm.λ₀` and `icm.Λ₀` are set according to NPMLE of `icm.S₀`. `icm.β` are set to 
zero; `icm.η` and `icm.expη` are set accordingly.
"""
function npmle!(
    icm     :: IntervalCensoredModel{T};
    maxiter :: Integer = 100_000,
    reltol  :: Real = 1e-8,
    verbose :: Bool = false
    ) where T <: Real
    n, m = length(icm.L), length(icm.ts)
    # construct membership matrix A = (αᵢⱼ), αᵢⱼ = I(sⱼ ∈ (Lᵢ, Rⱼ])
    A = Matrix{T}([icm.L[i] < icm.ts[j] ≤ icm.R[i] for i in 1:n, j in 1:m])
    # scratch space
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    # starting point (1/m, ..., 1/m)
    p = fill(T(1 / m), m)
    logl = sum(log, mul!(storage_n, A, p))
    verbose && println("iter=0, logl=$logl")
    for iter in 1:maxiter
        logl_prev = logl
        storage_n .= inv.(storage_n)
        mul!(storage_m, transpose(A), storage_n)
        p .*= storage_m ./ n
        logl = sum(log, mul!(storage_n, A, p))
        verbose && println("iter=$iter, logl=$logl")
        # check convergence criterion
        if abs(logl - logl_prev) < reltol * (abs(logl_prev) + 1)
            break
        elseif iter == maxiter
            @warn "maximum iterations $maxiter reached"
        end
    end
    # survival function
    icm.S₀[end] = 0
    for i in m-1:-1:1
        icm.S₀[i] = icm.S₀[i+1] + p[i+1]
    end
    # hazard function 
    icm.λ₀[1] = p[1]
    for i in 2:m
        icm.λ₀[i] = p[i] / icm.S₀[i - 1]
    end
    # cumulative hazard
    cumsum!(icm.Λ₀, icm.λ₀)
    # set β to 0
    fill!(icm.β, 0)
    fill!(icm.η, 0)
    fill!(icm.expη, 1)
    # return
    icm
end

function fit!(
    icm      :: IntervalCensoredModel,
    solver    = Ipopt.IpoptSolver(print_level=0);
    init     :: IntervalCensoredModel = icm,
    verbose  :: Bool = true
    )
    # set up NLP optimization problem
    m, p = length(icm.ts), size(icm.Z, 2)
    npar = m + p
    optm = MathProgBase.NonlinearModel(solver)
    lb   = [fill(0.0, m); fill(-Inf, p)]
    ub   = fill(Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, icm)
    par0 = Vector{Float64}(undef, npar)
    modelpar_to_optimpar!(par0, icm)
    MathProgBase.setwarmstart!(optm, par0)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # refresh objective, gradient, and Hessian
    optimpar_to_modelpar!(icm, MathProgBase.getsolution(optm))
    icm.S₀ .= exp.(- icm.Λ₀)
    loglikelihood!(icm, true, true)
    # inference for β
    idx = icm.λ₀ .> 1e-6
    Hλ₀λ₀_eval, Hλ₀λ₀_evec = eigen(Symmetric(icm.Hλ₀λ₀[idx, idx]))
    for (i, e) in enumerate(Hλ₀λ₀_eval)
        Hλ₀λ₀_eval[i] = e < 0 ? 0 : inv(e)
    end
    icm.Vββ .= icm.Hββ - 
        transpose(icm.Hλ₀β[idx, :]) * (Hλ₀λ₀_evec * 
        lmul!(Diagonal(Hλ₀λ₀_eval), transpose(Hλ₀λ₀_evec) * icm.Hλ₀β[idx, :]))
    Vββ_eval, Vββ_evec = eigen(Symmetric(icm.Vββ))
    for (i, e) in enumerate(Vββ_eval)
        Vββ_eval[i] = e < 0 ? 0 : inv(e)
    end
    icm.Vββ .= Vββ_evec * Diagonal(Vββ_eval) * transpose(Vββ_evec)
    icm.isfitted[1] = true
    icm
end

"""
    modelpar_to_optimpar!(par, icm)

Translate model parameters in `icm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    icm :: IntervalCensoredModel
    )
    m, p = length(icm.ts), size(icm.Z, 2)
    copyto!(par, icm.λ₀)
    copyto!(par, m + 1, icm.β, 1, p)
    par
end

"""
    optimpar_to_modelpar!(icm, par)

Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    icm :: IntervalCensoredModel, 
    par :: Vector
    )
    m, p = length(icm.ts), size(icm.Z, 2)
    copyto!(icm.λ₀, 1, par, 1, m)
    for (i, λi) in enumerate(icm.λ₀)
        (icm.λ₀[i] < 0) && (icm.λ₀[i] = 0)
    end
    cumsum!(icm.Λ₀, icm.λ₀)
    copyto!(icm.β, 1, par, m + 1, p)
    mul!(icm.η, icm.Z, icm.β)
    icm.expη .= exp.(icm.η)
    icm
end

function MathProgBase.initialize(
                       :: IntervalCensoredModel, 
    requested_features :: Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(::IntervalCensoredModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
    icm :: IntervalCensoredModel, 
    par :: Vector
    )
    optimpar_to_modelpar!(icm, par)
    # println(loglikelihood!(icm))
    loglikelihood!(icm)
end

function MathProgBase.eval_grad_f(
    icm  :: IntervalCensoredModel,
    grad :: Vector, 
    par  :: Vector
    )
    m, p = length(icm.ts), size(icm.Z, 2)
    optimpar_to_modelpar!(icm, par) 
    loglikelihood!(icm, true)
    # gradient wrt λ₀
    copyto!(grad, icm.∇λ₀)
    # gradient wrt β
    copyto!(grad, m + 1, icm.∇β, 1, p)
    nothing
end

MathProgBase.eval_g(icm::IntervalCensoredModel, g, par) = nothing
MathProgBase.jac_structure(icm::IntervalCensoredModel) = Int[], Int[]
MathProgBase.eval_jac_g(icm::IntervalCensoredModel, J, par) = nothing

function MathProgBase.hesslag_structure(icm::IntervalCensoredModel)
    # our Hessian is a dense (m+p)-by-(m+p) matrix
    m, p = length(icm.ts), size(icm.Z, 2)
    arr1 = Vector{Int}(undef, ◺(m + p))
    arr2 = Vector{Int}(undef, ◺(m + p))
    idx  = 1
    @inbounds for j in 1:(m + p), i in 1:j
        arr1[idx] = i
        arr2[idx] = j
        idx      += 1
    end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
    icm :: IntervalCensoredModel, 
    H   :: Vector{T},
    par :: Vector{T}, 
    σ   :: T, 
    μ   :: Vector{T}
    ) where {T}
    m, p = length(icm.ts), size(icm.Z, 2)
    # refresh obj, gradient, and hessian
    optimpar_to_modelpar!(icm, par)
    loglikelihood!(icm, true, true)
    # Hλ₀λ₀
    idx = 1
    @inbounds for j in 1:m, i in 1:j
        H[idx] = -icm.Hλ₀λ₀[i, j]
        idx   += 1
    end
    # Hλ₀β and Hββ
    @inbounds for j in 1:p
        for i in 1:m
            H[idx] = -icm.Hλ₀β[i, j]
            idx   += 1
        end
        for i in 1:j
            H[idx] = -icm.Hββ[i, j]
            idx   += 1
        end
    end
    lmul!(σ, H)
end

"""
    ◺(n::Integer)

Triangular number `n * (n + 1) / 2`.
"""
@inline ◺(n::Integer) = (n * (n + 1)) >> 1
