const SUSCEPTIBLE = 1 
const INFECTIOUS = 2

#compute derivatives of the log-likelihood with respect to infection rates of incoming edges
function der_λ(bp::MPBP{G,F}, i::Integer, ::Type{U}; svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
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
        lz = normalize!(B12)
        any(any(isnan, b) for b in B12) && @error "NaN in tensor train"
        compress!(B12; svd_trunc)
        any(any(isnan, b) for b in B12) && @error "NaN in tensor train"
        B12, lz + lz1 + lz2, d1 + d2
    end

    C, full, logzᵢ, sumlogzᵢ₂ⱼ, B, logzs = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)

    Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bᵢ = Bᵢ |> mpem2 |> marginalize
    logzᵢ += normalize!(bᵢ)
    # zᵢ = exp(logzᵢ + normalize!(bᵢ))

    λder = zeros(length(ein))
    for j in eachindex(ein)
        μⱼᵢ, ψᵢⱼ = μin[j], ψout[j]
        (Bj,logzj,dj) = B[j]
        
        # logder = -Inf
        der = 0.0
        for s in 1:T
            μⱼᵢˢ, ψᵢⱼˢ = μⱼᵢ[s], ψᵢⱼ[s]
            Bjs = Bj[s]
            Bjsold = copy(Bjs)
            for state in [INFECTIOUS, SUSCEPTIBLE]
                @tullio Bjs[m,n,yⱼ,xᵢ] = (yⱼ==state) * ψᵢⱼˢ[xᵢ,$INFECTIOUS] *  μⱼᵢˢ[m,n,$INFECTIOUS,xᵢ]

                full, logz = op((C[j], logzs[j], dᵢ-1), (Bj, 0.0, 1))
                b = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ) |> mpem2
                normb = normalization(b)
                logz += log(max(0.0,normb))
                der += (2*state-3)*exp(logz)
            end
            Bj[s] = Bjsold
        end

        λder[j] = sign(der) * exp(log(abs(der)) - logzᵢ)
        # λder[j] = der/zᵢ
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
    logzᵢ += normalize!(bᵢ)
    # zᵢ = exp(logzᵢ + normalize!(bᵢ))

    ρder = 0.0
    for s in 1:T
        q = length(ϕᵢ[1])
        B = [zeros(size(a,1), size(a,2), q, 1, q) for a in full]   # can remove the qj=1 (and also all dependences on xⱼᵗ afterwards)?
        for t in 1:T
            fullᵗ,Bᵗ = full[t], B[t]
            W = zeros(q,q,1,size(fullᵗ,3))
            if t==s
                @tullio avx=false W[xᵢᵗ⁺¹,INFECTIOUS,xⱼᵗ,yᵗ] = ((xᵢᵗ⁺¹==SUSCEPTIBLE)-(xᵢᵗ⁺¹==INFECTIOUS)) * ϕᵢ[$t][INFECTIOUS]
            else
                @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob_y(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,yᵗ,1) * ϕᵢ[$t][xᵢᵗ]
            end
            @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * fullᵗ[m,n,yᵗ,xᵢᵗ]
        end
        fullᵀ,Bᵀ = full[end], B[end]
        @tullio Bᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = fullᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
        any(any(isnan, b) for b in B) && @error "NaN in tensor train"

        b = MPEM3(B) |> mpem2
        ρder += normalization(b)*exp(logzᵢ)
    end

    return sign(ρder) * exp(log(abs(ρder)) - logzᵢ)
    # return ρder/zᵢ
end


# performs one step of Gradient Ascent for parameters of node i (i.e. infection rates of incoming edges and recovery rate)
function stepga!(bp::MPBP{G,F}, i::Integer, λstep::F=1e-2, ρstep::F=1e-2; method::Integer=1, svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph,F<:Real,U<:RecursiveBPFactor}
    @unpack g, w = bp
    ein = inedges(g,i)
    wᵢ, dᵢ  = w[i], length(ein)
    # @assert wᵢ[1] isa U

    λder = der_λ(bp, i, eltype(bp.w[1]); svd_trunc)
    ρder = der_ρ(bp, i, eltype(bp.w[1]); svd_trunc)
    
    if method==1
        for j in 1:dᵢ
            wᵢ[1].λ[j] += λstep*λder[j]
        end
        wᵢ[1].ρ += ρstep*ρder

    elseif method==2
        for j in 1:dᵢ
            wᵢ[1].λ[j] *= (1 + λstep*sign(λder[j]))
        end
        wᵢ[1].ρ *= (1 + ρstep*sign(ρder))

    elseif method==3
        for j in 1:dᵢ
            wᵢ[1].λ[j] += λstep*sign(λder[j])
        end
        wᵢ[1].ρ += ρstep*sign(ρder)
    end

    for j in 1:dᵢ
        if wᵢ[1].λ[j] < 0
            wᵢ[1].λ[j] = 0+1e-6
        end
        # wᵢ[1].λ[j] < 0 && wᵢ[1].λ[j] = 0+1e-6
        if wᵢ[1].λ[j] > 1
            wᵢ[1].λ[j] = 1-1e-6
        end
        # wᵢ[1].λ[j] > 1 && wᵢ[1].λ[j] = 1-1e-6
    end
    if wᵢ[1].ρ < 0
        wᵢ[1].ρ = 0+1e-6
    end
    # wᵢ[1].ρ[j] < 0 && wᵢ[1].ρ[j] = 0+1e-6
    if wᵢ[1].ρ > 1
        wᵢ[1].ρ = 1-1e-6
    end
    # wᵢ[1].ρ[j] > 1 && wᵢ[1].ρ[j] = 1-1e-6

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

function save_data(bp::MPBP; sites=vertices(bp.g)) where {F}
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

function inference_parameters(bp::MPBP; method::Integer=1, maxiter::Integer=5, λstep=1e-2, ρstep=1e-2, svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress=true, tol=1e-10, nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, cb_inf=CB_INF(bp;showprogress))

    for iter in 1:maxiter
        for i in nodes
            onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc, damp=0.0)     # why damp?
        end

        for i in nodes
            stepga!(bp, i, λstep, ρstep; method, svd_trunc)
        end

        Δ = cb_inf(bp,iter,svd_trunc)
        Δ < tol && return iter, cb_inf
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end

    return maxiter, cb_inf
end