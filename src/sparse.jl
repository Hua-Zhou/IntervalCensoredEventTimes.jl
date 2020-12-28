struct IntervalCensoredSparseModel{T <: BlasReal}
    basemodel :: IntervalCensoredModel{T} # low-dimensional base model
    X         :: AbstractMatrix{T}        # predictors for variable selection
    βx        :: Vector{T}                # regression coefficients for X
    k         :: Int                      # ||βₓ||₀ = number of nonzero coefficients
    # working arrays
    ∇βx       :: Vector{T}
    ηx        :: Vector{T}
    storage_m :: Vector{T}
    storage_px :: Vector{T}
    storage_pz :: Vector{T}
end

function IntervalCensoredSparseModel(
    basemodel :: IntervalCensoredModel{T},
    X         :: AbstractMatrix{T},
    k         :: Integer
    ) where {T <: BlasReal}
    n, m = length(basemodel.C), length(basemodel.ts)
    px, pz = size(X, 2), length(basemodel.β)
    @assert size(X, 1) == n "rows of X does not match that of Z"
    βx         = Vector{T}(undef, px)
    ∇βx        = Vector{T}(undef, px)
    ηx         = Vector{T}(undef, n)
    storage_m  = Vector{T}(undef, m)
    storage_px = Vector{T}(undef, px)
    storage_pz = Vector{T}(undef, pz)
    # constructor
    IntervalCensoredSparseModel{T}(
        basemodel, X, βx, k, 
        ∇βx, ηx, storage_m, storage_px, storage_pz)
end

function iht!(
    icsm    :: IntervalCensoredSparseModel{T};
    maxiter :: Integer = 100_000,
    reltol  :: Real = 1e-4,
    verbose :: Bool = false,
    printfreq :: Integer = 10,
    debiasing :: Bool = false) where T <: Real
    # fit the base model
    @info "Fit the base model:"
    icsm.basemodel.isfitted[1] || fit!(icsm.basemodel)
    display(icsm.basemodel)
    # ICM for debiasing model
    n, pz, px = length(icsm.basemodel.C), length(icsm.basemodel.β), length(icsm.βx)
    if debiasing
        icm_debias = IntervalCensoredModel(
            [icsm.basemodel.Z zeros(T, n, icsm.k)],
            icsm.basemodel.L,
            icsm.basemodel.R)
    end
    # start βₓ from all 0s
    fill!(icsm.βx, 0)
    logl = loglikelihood!(icsm.basemodel, true, true)
    verbose && @info("IHT iterations:")
    verbose && println("iter=0, logl=$logl")
    for iter in 1:maxiter
        # update λ₀
        stepsize = abs2(norm(icsm.basemodel.∇λ₀)) / dot(icsm.basemodel.∇λ₀, 
        mul!(icsm.storage_m, icsm.basemodel.Hλ₀λ₀, icsm.basemodel.∇λ₀))
        for (j, λ₀j) in enumerate(icsm.basemodel.λ₀)
            icsm.basemodel.λ₀[j] = 
            max(icsm.basemodel.λ₀[j] + stepsize * icsm.basemodel.∇λ₀[j], 0)
        end
        cumsum!(icsm.basemodel.Λ₀, icsm.basemodel.λ₀)
        # update βz
        stepsize = abs2(norm(icsm.basemodel.∇β)) / dot(icsm.basemodel.∇β, 
        mul!(icsm.storage_pz, icsm.basemodel.Hββ, icsm.basemodel.∇β))
        icsm.basemodel.β .+= stepsize .* icsm.basemodel.∇β
        mul!(icsm.basemodel.η, icsm.basemodel.Z, icsm.basemodel.β)
        # update βx
        mul!(icsm.∇βx, transpose(icsm.X), icsm.basemodel.βres)
        mul!(icsm.basemodel.storage_n, icsm.X, icsm.∇βx)
        icsm.basemodel.storage_n .*= icsm.basemodel.glmwt
        stepsize = abs2(norm(icsm.∇βx) / norm(icsm.basemodel.storage_n))
        icsm.βx .+= stepsize .* icsm.∇βx
        # project βx to set sparse set {||βx||₀ ≤ k}
        icsm.storage_px .= abs.(icsm.βx)
        pivot = partialsort!(icsm.storage_px, icsm.k, rev=true)
        for (j, βxj) in enumerate(icsm.βx)
            if abs(βxj) < pivot; icsm.βx[j] = 0; end
        end
        # debiasing step
        βx_sp = sparsevec(icsm.βx)
        if debiasing
            # warm start debiased model
            copyto!(icm_debias.β, icsm.basemodel.β)
            copyto!(icm_debias.β, pz + 1, βx_sp.nzval, 1, icsm.k)
            @views icm_debias.Z[:, pz+1:end] .= icsm.X[:, βx_sp.nzind]
            mul!(icm_debias.η, icm_debias.Z, icm_debias.β)
            icm_debias.expη .= exp.(icm_debias.η)
            copyto!(icm_debias.λ₀, icsm.basemodel.λ₀)
            copyto!(icm_debias.Λ₀, icsm.basemodel.Λ₀)
            # fit debiased model
            fit!(icm_debias)
            # copy debiased estimates back
            copyto!(icsm.basemodel.β, 1, icm_debias.β, 1, pz)
            copyto!(βx_sp.nzval, 1, icm_debias.β, pz + 1, icsm.k)
            icsm.βx .= Vector{T}(βx_sp)
            copyto!(icsm.basemodel.η, icm_debias.η)
            copyto!(icsm.basemodel.expη, icm_debias.expη)            
            copyto!(icsm.basemodel.λ₀, icm_debias.λ₀)
            copyto!(icsm.basemodel.Λ₀, icm_debias.Λ₀)
        else
            # update η = ηx + ηz
            icsm.basemodel.η   .+= mul!(icsm.ηx, icsm.X, βx_sp)
            icsm.basemodel.expη .= exp.(icsm.basemodel.η)
        end
        # update gradient and Hessian wrt λ₀, βz
        logl_old = logl
        logl = loglikelihood!(icsm.basemodel, true, true)
        verbose && iter%printfreq == 0 && println("iter=$iter, logl=$logl")
        # check convergence
        if abs(logl - logl_old) < reltol * (abs(logl_old) + 1)
            break
        elseif iter == maxiter
            @warn "maximum iterations reached!"
        end
    end
    icsm
end
