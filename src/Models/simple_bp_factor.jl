# for a `SimpleBPFactor`, outgoing messages can be computed recursively
abstract type SimpleBPFactor <: BPFactor; end

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q,T,F}, 
        wᵢ::Vector{U}, ψᵢₗ, l::Integer) where {q,T,F,U<:SimpleBPFactor}
    @assert q == 2
    map(1:T+1) do t
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        qy = nstates(U, l)
        AAᵗ = zeros(size(Aᵗ, 1), size(Aᵗ, 2), qy, qy)
        if t ≤ T
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = prob_partial_msg(wᵢ[$t],yₗᵗ,yₗ₁ᵗ,xₗᵗ,l) * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        else
            @tullio AAᵗ[m,n,yₗᵗ,xᵢᵗ] = 1/q * Aᵗ[m,n,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] * ψᵢₗ[$t][xᵢᵗ,xₗᵗ]
        end
    end |> MPEM2
end


# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij(A::MPEM2{q,T,F}, wᵢ::Vector{U}, ϕᵢ, 
    d::Integer; prob = prob_ijy) where {q,T,F,U<:SimpleBPFactor}
    B = [zeros(q, q, size(a,1), size(a,2), q) for a in A]
    for t in 1:T
        Aᵗ,Bᵗ = A[t], B[t]
        @tullio Bᵗ[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[xᵢᵀ,xⱼᵀ,m,n,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && println("NaN in tensor train")
    return MPEM3(B)
end

function f_bp(A::Vector{MPEM2{q,T,F}},
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}},
        j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F,U<:SimpleBPFactor}

    d = length(A) - 1   # number of neighbors other than j
    @assert j ∈ eachindex(A)

    # initialize recursion
    M = reshape(vcat(ones(1,q), zeros(q-1,q)), (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    logz = 0.0
    l = 1
    for k in eachindex(A)
        k == j && continue
        mᵢⱼₗ₁ = f_bp_partial(A[k], mᵢⱼₗ₁, wᵢ, ψₙᵢ[k], l)
        logz +=  normalize!(mᵢⱼₗ₁)
        l += 1
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, wᵢ, ϕᵢ, d; prob = prob_ijy)

    return B, logz
end

function f_bp_dummy_neighbor(A::Vector{MPEM2{q,T,F}}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        svd_trunc=TruncThresh(1e-6)) where {q,T,F,U<:SimpleBPFactor}
    
    d = length(A)

    # initialize recursion
    M = reshape(vcat(ones(1,q), zeros(q-1,q)), (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    logz = 0.0
    # compute partial messages from all neighbors
    for l in eachindex(A)
        mᵢⱼₗ₁ = f_bp_partial(A[l], mᵢⱼₗ₁, wᵢ, ψₙᵢ[l], l)
        logz +=  normalize!(mᵢⱼₗ₁)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij(mᵢⱼₗ₁, wᵢ, ϕᵢ, d; prob = prob_ijy_dummy)

    return B, logz
end


function beliefs(bp::MPBP{q,T,F,<:SimpleBPFactor};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    b = [[zeros(q) for _ in 0:T] for _ in vertices(bp.g)]
    for i in eachindex(b)
        A = onebpiter_dummy_neighbor(bp, i; svd_trunc)
        b[i] .= firstvar_marginal(A)
    end
    b
end

function beliefs_tu(bp::MPBP{q,T,F,<:SimpleBPFactor};
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    b = [[zeros(q, q) for _ in 0:T, _ in 0:T] for _ in vertices(bp.g)]
    for i in eachindex(b)
        A = onebpiter_dummy_neighbor(bp, i; svd_trunc)
        b[i] .= firstvar_marginal_tu(A)
    end
    b
end

### INFINITE REGULAR GRAPHS

function onebpiter_infinite_graph(A::MPEM2, k::Integer, wᵢ::Vector{U}, 
        ϕᵢ, ψₙᵢ;
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:SimpleBPFactor}

    B, _ = f_bp(fill(A, k), wᵢ, ϕᵢ, ψₙᵢ, 1)
    C = mpem2(B)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    A_new
end

function iterate_bp_infinite_graph(T::Integer, k::Integer, wᵢ::Vector{U},
        ϕᵢ = fill(ones(getq(U)), T+1);
        ψₙᵢ = fill(fill(ones(getq(U), getq(U)), T+1), k),
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true) where {U<:SimpleBPFactor}
    @assert length(ϕᵢ) == T + 1
    @assert length(wᵢ) == T
    
    A = mpem2(getq(U), T)
    Δs = fill(NaN, maxiter)
    m = firstvar_marginal(A)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(maxiter; dt, desc="Iterating BP on infinite graph")
    for it in 1:maxiter
        A = onebpiter_infinite_graph(A, k, wᵢ, ϕᵢ, ψₙᵢ; svd_trunc)
        m_new = firstvar_marginal(A)
        Δ = maximum(abs, bb_new[1] - bb[1] for (bb_new, bb) in zip(m_new, m))
        Δs[it] = Δ
        Δ < tol && return A, it, Δs
        m, m_new = m_new, m
        rounded_Δ = round(Δ, digits=ceil(Int,abs(log(tol))))
        next!(prog, showvalues=[(:iter, "$it/$maxiter"), (:Δ,"$rounded_Δ/$tol")])
    end
    A, maxiter, Δs
end

function onebpiter_dummy_infinite_graph(A::MPEM2, k::Integer,
        wᵢ::Vector{U}, ϕᵢ, ψₙᵢ; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:SimpleBPFactor}

    B, _ = f_bp_dummy_neighbor(fill(A, k), wᵢ, ϕᵢ, ψₙᵢ)
    C = mpem2(B)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    A_new
end

# A is the message already converged
# return marginals, expectations of marginals and covariances
function observables_infinite_graph(A::MPEM2, k::Integer, 
        wᵢ::Vector{<:U}, ϕᵢ;
        ψₙᵢ = fill(fill(ones(getq(U),getq(U)), length(A)), k),
        svd_trunc::SVDTrunc=TruncThresh(1e-6), 
        showprogress=true) where {U<:SimpleBPFactor}

    Anew = onebpiter_dummy_infinite_graph(A, k, wᵢ, ϕᵢ, ψₙᵢ; svd_trunc)
    b = firstvar_marginal(Anew)
    b_tu = firstvar_marginal_tu(Anew; showprogress)
    r = marginal_to_expectation.(b_tu, (U,))
    m = marginal_to_expectation.(b, (U,))
    c = MatrixProductBP.covariance(r, m)
    b, m, c
end