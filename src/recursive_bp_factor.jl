const SUSCEPTIBLE = 1 
const INFECTIOUS = 2


""""
For a `w::U` where `U<:RecursiveBPFactor`, outgoing messages can be computed recursively
A `<:RecursiveBPFactor` must implement: `nstates`, `prob_y`, `prob_xy` and `prob_yy`
Optionally, it can also implement `prob_y_partial` and `(w::U)(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)`
"""
abstract type RecursiveBPFactor <: BPFactor; end

#### the next five methods are the minimal needed interface for a new <:RecursiveBPFactor

"Number of states for aux variable which accumulates the first `l` neighbors"
nstates(w::RecursiveBPFactor, l::Integer) = error("Not implemented")

"P(xᵢᵗ⁺¹|xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ)
Might depend on the degree `dᵢ` because of a change of variable from 
    y ∈ {1,2,...} to its physical value, e.g. {-dᵢ,...,dᵢ} for Ising"
prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U<:RecursiveBPFactor} = error("Not implemented")

"P(yₖᵗ| xₖᵗ, xᵢᵗ)
`k` is the index of the current neighbor among the set of neighbors"
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ) = error("Not implemented")
# By default we assume no dependence on `k`. A case where `k` actually matters is e.g. IntegerGlauberFactor
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ, k) = prob_xy(wᵢ, yₖ, xₖ, xᵢ)

"P(yₐᵦ|yₐ,yᵦ,xᵢᵗ)"
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ, d1, d2) = prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ)
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ) = error("Not implemented")
prob_y0(wᵢ::RecursiveBPFactor, y, xᵢᵗ) = y == 1

##############################################
#### the next methods are optional


"P(xᵢᵗ⁺¹|xᵢᵗ, xₙᵢᵗ, d)"
function (wᵢ::RecursiveBPFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    d = length(xₙᵢᵗ)
    Pyy = [float(prob_y0(wᵢ, y, xᵢᵗ)) for y in 1:nstates(wᵢ,0)]
    for k in 1:d
        Pyy = [sum(prob_yy(wᵢ, y, y1, y2, xᵢᵗ, 1, k-1) *
                   prob_xy(wᵢ, y1, xₙᵢᵗ[k], xᵢᵗ, k) *
                   Pyy[y2]
                   for y1 in 1:nstates(wᵢ,1), y2 in eachindex(Pyy)) 
               for y in 1:nstates(wᵢ,k)]
    end
    sum(Pyy[y] * prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, y, d) for y in eachindex(Pyy))
end

"P(xᵢᵗ⁺¹|xᵢᵗ, yₙᵢnotk, xₖᵗ, d)"
function prob_y_partial(wᵢ::RecursiveBPFactor, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, k)
    sum(prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d + 1) * 
        prob_xy(wᵢ, y2, xₖᵗ, xᵢᵗ, k) * 
        prob_yy(wᵢ, yᵗ, y1, y2, xᵢᵗ, d, 1) 
        for yᵗ in 1:nstates(wᵢ, d + 1), y2 in 1:nstates(wᵢ,1))
end


#####################################################

function prob_y_dummy(wᵢ::RecursiveBPFactor, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, j)
    prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, y1, d)
end

# compute matrix B for mᵢⱼ
function f_bp_partial_ij(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer, qj, j) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_partial, qj, j)
end

# compute matrix B for bᵢ
function f_bp_partial_i(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_dummy, 1, 1)
end

function _f_bp_partial(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    B = [zeros(size(a,1), size(a,2), q, qj, q) for a in A]
    for t in Iterators.take(eachindex(A), length(A)-1)
        Aᵗ,Bᵗ = A[t], B[t]
        W = zeros(q,q,qj,size(Aᵗ,3))
        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j)*ϕᵢ[$t][xᵢᵗ]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ]*Aᵗ[m,n,yᵗ,xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && @error "NaN in tensor train"
    return MPEM3(B; z = A.z)
end

function _f_bp_partial(A::PeriodicMPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    B = [zeros(size(a,1), size(a,2), q, qj, q) for a in A]
    for t in eachindex(A)
        Aᵗ,Bᵗ = A[t], B[t]
        W = zeros(q,q,qj,size(Aᵗ,3))
        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j) * ϕᵢ[$t][xᵢᵗ]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * Aᵗ[m,n,yᵗ,xᵢᵗ]
    end
    any(any(isnan, b) for b in B) && @error "NaN in tensor train"
    return PeriodicMPEM3(B; z = A.z)
end

# compute ̃m{∂i∖j→i}(̅y_{∂i∖j},̅xᵢ)
function compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{M2}, ψout, T, svd_trunc) where {U<:RecursiveBPFactor, M2<:AbstractMPEM2}
    @debug @assert all(float(normalization(a)) ≈ 1 for a in μin) "$([float(normalization(a)) for a in μin])"
    B = map(eachindex(ψout)) do k
        Bk = map(zip(wᵢ, μin[k], ψout[k])) do (wᵢᵗ, μₖᵢᵗ, ψᵢₖᵗ)
            Pxy = zeros(nstates(wᵢᵗ,1), size(μₖᵢᵗ, 3), qi)
            @tullio avx=false Pxy[yₖ,xₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ,k) * ψᵢₖᵗ[xᵢ,xₖ]
            @tullio _[m,n,yₖ,xᵢ] := Pxy[yₖ,xₖ,xᵢ] * μₖᵢᵗ[m,n,xₖ,xᵢ]
        end |> M2
        Bk, 1
    end
    # now in B there are P(yₖᵗ|xₖᵗ) m_{k→i}(yₖᵗ,xᵢᵗ) ∀k,∀t  (written as MPS)

    function op((B1, d1), (B2, d2))
        BB = map(zip(wᵢ,B1,B2)) do (wᵢᵗ,B₁ᵗ,B₂ᵗ)
            Pyy = zeros(nstates(wᵢᵗ,d1+d2), size(B₁ᵗ,3), size(B₂ᵗ,3), size(B₁ᵗ,4))
            @tullio avx=false Pyy[y,y1,y2,xᵢ] = prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2) 
            @tullio B3[m1,m2,n1,n2,y,xᵢ] := Pyy[y,y1,y2,xᵢ] * B₁ᵗ[m1,n1,y1,xᵢ] * B₂ᵗ[m2,n2,y2,xᵢ]
            @cast _[(m1,m2),(n1,n2),y,xᵢ] := B3[m1,m2,n1,n2,y,xᵢ]
        end
        B = M2(BB; z = B1.z * B2.z)
        any(any(isnan, b) for b in B) && @error "NaN in tensor train"
        compress!(B; svd_trunc)
        normalize_eachmatrix!(B)    # keep this one?
        any(any(isnan, b) for b in B) && @error "NaN in tensor train"
        B, d1 + d2
    end
    
    Minit = [[float(prob_y0(wᵢ[t], y, xᵢ)) for _ in 1:1,
                _ in 1:1,
                y in 1:nstates(wᵢ[t],0),
                xᵢ in 1:qi]
            for t=1:T+1]
    init = (M2(Minit), 0)
    dest, (full,)  = cavity(B, op, init)
    (C,) = unzip(dest)
    C, full
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{G,F}, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6), damp::Real=0.0) where {G<:AbstractIndexedDiGraph,F<:Real,U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    @assert wᵢ[1] isa U
    C, full = compute_prob_ys(wᵢ, nstates(bp,i), μ[ein.|>idx], ψ[eout.|>idx], getT(bp), svd_trunc)
    sumlogzᵢ₂ⱼ = zero(F)
    for (j,e) = enumerate(eout)
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ - 1, nstates(bp, dst(e)), j)
        μj = orthogonalize_right!(mpem2(B); svd_trunc)
        sumlogzᵢ₂ⱼ += set_msg!(bp, μj, idx(e), damp, svd_trunc)
    end
    B = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bp.b[i] = B |> mpem2 |> marginalize
    logzᵢ = normalize!(bp.b[i])
    bp.f[i] = (dᵢ/2-1)*logzᵢ - (1/2)*sumlogzᵢ₂ⱼ
    # nothing
    return logzᵢ
end

#compute derivatives of the log-likelihood with respect to infection rates of incoming edges
function der_λ(bp::MPBP{G,F}, i::Integer, ::Type{U}; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
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

function der_ρ(bp::MPBP{G,F}, i::Integer, ::Type{U}; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    qi = nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    C, full, logzᵢ, sumlogzᵢ₂ⱼ, B, logzs = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)

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

# write message to destination after applying damping
function set_msg!(bp::MPBP{G,F,V,M2}, μj::M2, edge_id, damp, svd_trunc) where {G,F,V,M2}
    @assert 0 ≤ damp < 1
    μ_old = bp.μ[edge_id]
    logzᵢ₂ⱼ = normalize!(μj)
    if damp > 0 
        μj = _compose(x->x*damp/(1-damp), μj, μ_old)
        compress!(μj; svd_trunc)
        normalize!(μj)
    end
    bp.μ[edge_id] = μj
    logzᵢ₂ⱼ
end

# adds a further transition xᵢᵗ->xᵢᵗ⁺¹ with probability `p` and rescales all other
#  transitions by `1-p`. Does nothing for `p=0`
struct DampedFactor{T<:RecursiveBPFactor,F<:Real} <: RecursiveBPFactor
    w :: T      # factor
    p :: F      # probability of staying in previous state        
    function DampedFactor(w::T, p::F) where {T<:RecursiveBPFactor,F<:Real}
        @assert 0 ≤ p ≤ 1
        new{T,F}(w, p)
    end
end

nstates(w::DampedFactor{T}, l::Integer) where {T} = nstates(w.w, l)

@forward DampedFactor.w prob_xy, prob_yy

function (wᵢ::DampedFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    return (1-wᵢ.p)*(wᵢ.w(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) + wᵢ.p*(xᵢᵗ⁺¹ == xᵢᵗ)  
end

function prob_y(wᵢ::DampedFactor, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)
    return (1-wᵢ.p)*(prob_y(wᵢ.w, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d)) + wᵢ.p*(xᵢᵗ⁺¹ == xᵢᵗ)
end

prob_y0(wᵢ::DampedFactor, y, xᵢᵗ) = prob_y0(wᵢ.w, y, xᵢᵗ)
