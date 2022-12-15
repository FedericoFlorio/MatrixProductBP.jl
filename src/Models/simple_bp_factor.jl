""""
For a `w::U` where `U<:RecursiveBPFactor`, outgoing messages can be computed recursively
A `<:RecursiveBPFactor` must implement: `nstates`, `prob_ijy_dummy`, `prob_xy` and `prob_yy`
Optionally, it can also implement `prob_ijy` and `(w::U)(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)`
"""
abstract type RecursiveBPFactor <: BPFactor; end

#### the next four methods are the minimal needed interface for a new <:RecursiveBPFactor

"Number of states for aux variable which accumulates the first `l` neighbors"
nstates(::Type{<:RecursiveBPFactor}, l::Integer) = error("Not implemented")

"P(xᵢᵗ⁺¹|xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ)"
prob_ijy_dummy(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ) where {U<:RecursiveBPFactor} = error("Not implemented")

"P(yₖᵗ, xₖᵗ, xᵢᵗ)"
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ) = error("Not implemented")

"P(yₐᵦ|yₐ,yᵦ,xᵢᵗ)"
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ) = error("Not implemented")


##############################################
#### the next two methods are optional

"P(xᵢᵗ⁺¹|xᵢᵗ, xₙᵢᵗ, d)"
function (wᵢ::RecursiveBPFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
    xᵢᵗ::Integer)
    U = typeof(wᵢ)
    d = length(xₙᵢᵗ)
    @assert all(x ∈ 1:nstates(U) for x in xₙᵢᵗ)
    Pyy = fill(1.0, 1)
    for k in 1:d
        Pyy = [sum(prob_yy(wᵢ, y, y1, y2, xᵢᵗ)*prob_xy(wᵢ, y1, xₙᵢᵗ[k], xᵢᵗ)*Pyy[y2]
                   for y1 in 1:nstates(U,1), y2 in 1:nstates(U,k-1)) 
               for y in 1:nstates(U,k)]
    end
    sum(Pyy[y] * prob_ijy_dummy(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, 1, y, d) for y in eachindex(Pyy))
end

function prob_ijy(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d) where {U<:RecursiveBPFactor}
    sum(prob_ijy_dummy(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, nothing, yᵗ, d + 1) * 
        prob_xy(wᵢ, y2, xₖᵗ, xᵢᵗ) * 
        prob_yy(wᵢ, yᵗ, y1, y2, xᵢᵗ) 
        for yᵗ in 1:nstates(U, d + 1), y2 in 1:nstates(U,1))
end


#####################################################


# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial(mₗᵢ::MPEM2, mᵢⱼₗ₁::MPEM2, 
        wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {U<:RecursiveBPFactor}
    T = getT(mₗᵢ)
    @assert getT(mᵢⱼₗ₁) == T
    map(1:T+1) do t
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        qxᵢ = nstates(U); qy = nstates(U, l)
        AAᵗ = zeros(size(Aᵗ, 1), size(Aᵗ, 2), qy, qxᵢ)        
        @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg(wᵢ[$t],yₗᵗ,yₗ₁ᵗ,xₗᵗ,l) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
    end |> MPEM2
end


# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
    d::Integer; prob = prob_ijy) where {U<:RecursiveBPFactor}
    q = nstates(U)
    B = [zeros(q, q, size(a,1), size(a,2), q) for a in A]
    for t in 1:getT(A)
        Aᵗ,Bᵗ = A[t], B[t]
        @tullio Bᵗ[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[xᵢᵀ,xⱼᵀ,m,n,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && println("NaN in tensor train")
    return MPEM3(B)
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{G,F,U}, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {G<:AbstractIndexedDiGraph,F<:Real,U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    dᵢ = length(ein)
    wᵢ, ϕᵢ = w[i], ϕ[i]
    T = getT(bp)
    A, ψout = μ[ein.|>idx], ψ[eout.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
 
    B = map(1:dᵢ) do k
        Bk = map(1:getT(A[k]) + 1) do t
            Akt, wᵢᵗ, ψᵢₖᵗ = A[k][t], wᵢ[t], ψout[k][t]
            Bkt = zeros(size(Akt,1), size(Akt,2), nstates(U, 1), size(Akt,4))
            @tullio Bkt[m,n,yₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ) * Akt[m,n,xₖ,xᵢ] * ψᵢₖᵗ[xᵢ,xₖ]
        end |> MPEM2
        Bk, 0.0, 1
    end

    Minit = fill(1.0, 1, 1, 1, nstates(U))
    init = (MPEM2(fill(Minit, T + 1)), 0.0, 0)

    function op((B1, lz1, n1), (B2, lz2, n2))
        B = map(eachindex(B1.tensors)) do t
            Bᵗ = kron2(B1[t], B2[t])
            Bout = zeros(size(Bᵗ,1), size(Bᵗ,2), nstates(U,n1+n2), size(Bᵗ,3))
            @tullio Bout[m,n,y,xᵢ] = prob_yy(wᵢ[$t],y,y1,y2,xᵢ) * Bᵗ[m,n,xᵢ,y1,y2]
        end |> MPEM2
        lz = normalize!(B)
        sweep_LtoR!(B, svd_trunc=TruncThresh(0.0))
        sweep_RtoL!(B; svd_trunc)
        B, lz + lz1 + lz2, n1 + n2
    end

    dest, (full, _)  = cavity(B, op, init)
    (C, logzs) = unzip(dest)

    logzᵢ = sum(logzs)
    for (j,e) = enumerate(eout)
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ - 1; prob = prob_ijy)
        μ[idx(e)] = sweep_RtoL!(mpem2(B); svd_trunc)
        logzᵢ += normalize!(μ[idx(e)])
    end
    B = f_bp_partial_ij(full, wᵢ, ϕᵢ, dᵢ; prob = prob_ijy_dummy)
    bp.b[i] = B |> mpem2 |> marginalize
    return dᵢ == 0 ? 0.0 : logzᵢ / dᵢ
end


function beliefs(bp::MPBP{G,F,U}) where {G,F,U<:RecursiveBPFactor}
    [marginals(bi) for bi in bp.b]
end

function beliefs_tu(bp::MPBP{G,F,U}) where {G,F,U<:RecursiveBPFactor}
    [marginals_tu(bi) for bi in bp.b]
end