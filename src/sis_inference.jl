const SUSCEPTIBLE = 1
const INFECTIOUS = 2


# compute derivatives of the log-likelihood with respect to infection rates of incoming edges
function der_λ(bp::MPBP{G,F}, i::Integer, ::Type{U}; svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    M2 = eltype(μ)
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    qi = nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    function op((B1, lz1, d1), (B2, lz2, d2))
        B12 = map(zip(wᵢ,B1,B2)) do (wᵢᵗ,B₁ᵗ,B₂ᵗ)
            Pyy = zeros(nstates(wᵢᵗ,d1+d2), size(B₁ᵗ,3), size(B₂ᵗ,3), size(B₁ᵗ,4))
            @tullio avx=false Pyy[y,y1,y2,xᵢ] = prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2)
            @tullio B3[m1,m2,n1,n2,y,xᵢ] := Pyy[y,y1,y2,xᵢ] * B₁ᵗ[m1,n1,y1,xᵢ] * B₂ᵗ[m2,n2,y2,xᵢ]
            @cast _[(m1,m2),(n1,n2),y,xᵢ] := B3[m1,m2,n1,n2,y,xᵢ]
        end |> M2
        any(any(isnan, b) for b in B12) && @error "NaN in tensor train"
        compress!(B12; svd_trunc)
        lz = normalize!(B12)
        any(any(isnan, b) for b in B12) && @error "NaN in tensor train"
        B12, lz + lz1 + lz2, d1 + d2
    end

    C, full, logzᵢ, sumlogzᵢ₂ⱼ, B, logzs = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)

    Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bᵢ = Bᵢ |> mpem2 |> marginalize
    logzᵢ += normalize!(bᵢ)

    λder = zeros(length(ein))
    for j in eachindex(ein)
        μⱼᵢ, ψᵢⱼ = μin[j], ψout[j]
        (Bj,logzj,dj) = B[j]
        
        der = Logarithmic(0.0)
        for s in 1:T
            μⱼᵢˢ, ψᵢⱼˢ = μⱼᵢ[s], ψᵢⱼ[s]
            Bjs = Bj[s]
            Bjsold = copy(Bjs)

            @tullio Bjs[m,n,yⱼ,xᵢ] = ((yⱼ==INFECTIOUS)-(yⱼ==SUSCEPTIBLE)) * ψᵢⱼˢ[xᵢ,$INFECTIOUS] * μⱼᵢˢ[m,n,$INFECTIOUS,xᵢ]
            
            full, logz = op((C[j], logzs[j], dᵢ-1), (Bj, logzj, 1))
            b = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ) |> mpem2
            b.z *= exp(Logarithmic,-logz) * full.z
            normb = normalization(b)
            der += normb

            Bj[s] = Bjsold
        end

        λder[j] = der * exp(Logarithmic,-logzᵢ) + logpriorder(wᵢ[1].λ[j])
    end

    return λder
end

# computes derivatives of the log-likelihood with respect to recovery rate
function der_ρ(bp::MPBP{G,F}, i::Integer, ::Type{U}; svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    qi = nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    C, full, logzᵢ, sumlogzᵢ₂ⱼ, B, = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)

    Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bᵢ = Bᵢ |> mpem2 |> marginalize
    zᵢ = exp(Logarithmic, normalize!(bᵢ))

    q = length(ϕᵢ[1])
    A = [zeros(size(a,1), size(a,2), q, 1, q) for a in full]
    As = [zeros(size(a,1), size(a,2), q, 1, q) for a in full]
    for t in 1:T
        fullᵗ,Aᵗ,Asᵗ = full[t], A[t], As[t]
        W = zeros(q,q,1,size(fullᵗ,3))

        @tullio avx=false W[xᵢᵗ⁺¹,INFECTIOUS,xⱼᵗ,yᵗ] = ((xᵢᵗ⁺¹==SUSCEPTIBLE)-(xᵢᵗ⁺¹==INFECTIOUS)) * ϕᵢ[$t][INFECTIOUS]
        @tullio Asᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * fullᵗ[m,n,yᵗ,xᵢᵗ]

        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob_y(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,yᵗ,1) * ϕᵢ[$t][xᵢᵗ]
        @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * fullᵗ[m,n,yᵗ,xᵢᵗ]
    end
    fullᵀ,Aᵀ = full[end], A[end]
    @tullio Aᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = fullᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, a) for a in A) && @error "NaN in tensor train"
    any(any(isnan, a) for a in A) && @error "NaN in tensor train"

    ρder = Logarithmic(0.0)
    for s in 1:T
        B = [A[1:s-1]...,As[s],A[s+1:end]...]
        b = MPEM3(B) |> mpem2
        ρder += normalization(b)
    end

    return float(ρder/zᵢ)
end


# performs one step of Gradient Ascent for parameters of node i (i.e. infection rates of incoming edges and recovery rate)
function stepga!(bp::MPBP{G,F}, i::Integer, λstep::F=1e-2, ρstep::F=1e-2; method::Integer=1, svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0, progress::Float64=Inf) where {G<:AbstractIndexedDiGraph,F<:Real}
    @unpack g, w = bp
    ein = inedges(g,i)
    wᵢ, dᵢ  = w[i], length(ein)

    λder = der_λ(bp, i, eltype(bp.w[i]); svd_trunc, logpriorder)
    ρder = der_ρ(bp, i, eltype(bp.w[i]); svd_trunc, logpriorder)
    
    if method==1
        for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*λder[j]
            end
        wᵢ[t].ρ += ρstep*ρder
        end

    elseif method==2
        for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] *= (1 + λstep*sign(λder[j]))
            end
            wᵢ[t].ρ *= (1 + ρstep*sign(ρder))
        end

    elseif method==21
        for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] *= (1 + λstep*sign(λder[j])*5^(progress<.2))
            end
            wᵢ[t].ρ *= (1 + ρstep*sign(ρder)*5^(progress<.2))
        end

    elseif method==3
        for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*sign(λder[j])
            end
            wᵢ[t].ρ += ρstep*sign(ρder)
        end

    elseif method==31
        for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*sign(λder[j])*5^(progress<.25)
            end
            wᵢ[t].ρ += ρstep*sign(ρder)*5^(progress<.25)
        end

    else
        @error "Invalid method"
    end

    for t in eachindex(wᵢ)
        for j in 1:dᵢ
            wᵢ[t].λ[j] = clamp(wᵢ[1].λ[j], 1e-6, 1-1e-6)
        end
        wᵢ[t].ρ = clamp(wᵢ[1].ρ, 1e-6, 1-1e-6)
    end

    return nothing
end

# what follows is the implementation of a callback to save parameters during the iterations and print info
struct PARAMS{T}
    λ :: Vector{Vector{T}}
    ρ :: Vector{T}
    function PARAMS(λ::Vector{Vector{T}},ρ::Vector{T}) where {T<:Real}
        new{T}(λ,ρ)
    end
end

function save_data(bp::MPBP; sites=vertices(bp.g))
    @unpack w = bp
    λ = [similar(w[i][1].λ) for i in sites]
    ρ = zeros(length(sites))

    for i in sites
        wᵢ = w[i]
        λ[i] = [λⱼᵢ for λⱼᵢ in wᵢ[1].λ]
        ρ[i] = wᵢ[1].ρ
    end

    return PARAMS(λ, ρ)
end

struct CB_INF{TP<:ProgressUnknown, F, T}
    prog :: TP
    data :: T
    Δs   :: Vector{Float64}
    f    :: F

    function CB_INF(bp::MPBP; showprogress::Bool=true, f::F=save_data, info="") where F
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running Gradient Ascent: iter", dt=dt)
        TP = typeof(prog)
        data = [save_data(bp)]
        T = typeof(data)
        Δs = zeros(0)
        new{TP,F,T}(prog, data, Δs, f)
    end
end

function (cb_inf::CB_INF)(bp::MPBP, it::Integer, svd_trunc::SVDTrunc)
    err(old, new) = abs.(new .- old ./ old)
    data_new = save_data(bp)
    @unpack λ, ρ = data_new
    λnew, ρnew = λ, ρ
    data_old = cb_inf.data[end]
    @unpack λ, ρ = data_old
    λold, ρold = λ, ρ

    Δ = max(maximum(err(ρold,ρnew)), maximum(maximum(err(λold[i],λnew[i])) for i in eachindex(λold)))
    push!(cb_inf.Δs, Δ)
    push!(cb_inf.data, data_new)
    next!(cb_inf.prog, showvalues=[(:Δ,Δ), summary_compact(svd_trunc)])
    flush(stdout)
    return Δ
end

function inference_parameters!(bp::MPBP; method::Integer=1, maxiter::Integer=5, λstep=1e-2, ρstep=1e-2, svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0, showprogress=true, tol=1e-10, nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, cb_inf=CB_INF(bp;showprogress))

    for iter in 1:maxiter
        Threads.@threads for i in nodes
            onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc, damp=0.0)
            # jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/trace_error/bp_it$(iter)_node$(i)_bp.jld2"; bp)
        end

        Threads.@threads for i in nodes
            stepga!(bp, i, λstep, ρstep; method, svd_trunc, logpriorder, progress=iter/maxiter)
            # jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/trace_error/bp_it$(iter)_node$(i)_ga.jld2"; bp)
        end

        Δ = cb_inf(bp,iter,svd_trunc)
        Δ < tol && return iter, cb_inf
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end

    return maxiter, cb_inf
end