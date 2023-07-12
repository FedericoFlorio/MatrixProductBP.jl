"""
Factor for the factor graph of a model solvable with MPBP.

Any `BPFactor` subtype must implement:
- A functor that computes the Boltzmann contribution to the joint probability

That's it!

"""
abstract type BPFactor; end

# Factors are not collections (avoid confusion in TensorCast)
Base.broadcastable(b::BPFactor) = Ref(b)

# compute outgoing message as a function of the incoming ones
# A is a vector with all incoming messages. At index j_index there is m(j → i)
# ψᵢⱼ are the ones living on the outedges of node i
function f_bp(A::Vector{MPEM2{F}}, wᵢ::Vector{U}, 
        ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}}, j_index::Integer;
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0)) where {F,U<:BPFactor}
    T = length(A[1]) - 1
    @assert all(length(a) == T + 1 for a in A)
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1
    q = length(ϕᵢ[1])
    qj = size(ψₙᵢ[j_index][1],1)
    @assert all(length(ϕᵢᵗ) == q for ϕᵢᵗ in ϕᵢ)
    @assert j_index in eachindex(A)
    z = length(A)      # z = |∂i|
    x_neigs = Iterators.product((1:size(ψₙᵢ[k][1],2) for k=1:z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T + 1)
 
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    for t in 1:T+1
        # select incoming A's but not the j-th one
        Aᵗ = kron2([A[k][t] for k in eachindex(A)[Not(j_index)]]...)
        nrows, ncols = size(Aᵗ, 1), size(Aᵗ, 2)
        Bᵗ = zeros(nrows, ncols, q, qj, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xₙᵢᵗ in x_neigs
                    xⱼᵗ = xₙᵢᵗ[j_index]
                    xₙᵢ₋ⱼᵗ = xₙᵢᵗ[Not(j_index)]
                    Bᵗ[:, :, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹] .+= (t == T + 1 ? 1.0 : wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) *
                        Aᵗ[:, :, xᵢᵗ, xₙᵢ₋ⱼᵗ...] *
                        prod(ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in enumerate(xₙᵢᵗ) if k != j_index; init=1.0)
                end
            end
            Bᵗ[:, :, xᵢᵗ, :, :] *= ϕᵢ[t][xᵢᵗ]
        end
        B[t] = Bᵗ
        any(isnan, Bᵗ) && @error "NaN in tensor at time $t"
        next!(prog, showvalues=[(:t, "$t/$T")])
    end

    return MPEM3(B), 0.0
end

# compute outgoing message to dummy neighbor to get the belief
function f_bp_dummy_neighbor(A::Vector{MPEM2{F}}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0)) where {F,U<:BPFactor}
    T = length(A[1]) - 1
    @assert all(length(a) == T + 1 for a in A)
    q = length(ϕᵢ[1])
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1
    z = length(A)      # z = |∂i|
    xₙᵢ = Iterators.product((1:size(ψₙᵢ[k][1],2) for k=1:z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T + 1)
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    for t in 1:T+1
        Aᵗ = kron2([A[k][t] for k in eachindex(A)]...)
        nrows = size(Aᵗ, 1)
        ncols = size(Aᵗ, 2)
        Bᵗ = zeros(nrows, ncols, q, 1, q)
        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xₙᵢᵗ in xₙᵢ
                    Bᵗ[:, :, xᵢᵗ, 1, xᵢᵗ⁺¹] .+= (t == T + 1 ? 1.0 : wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)) .*
                                                    Aᵗ[:, :, xᵢᵗ, xₙᵢᵗ...] .* ϕᵢ[t][xᵢᵗ] .*
                                                    prod(ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in enumerate(xₙᵢᵗ))
                end
            end
        end
        B[t] = Bᵗ
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        next!(prog, showvalues=[(:t, "$t/$T")])
    end

    return MPEM3(B), 0.0
end

function pair_belief_as_mpem(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)
    A = map(zip(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)) do (Aᵢⱼᵗ, Aⱼᵢᵗ, ψᵢⱼᵗ)
        @cast Aᵗ[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * 
            Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ]
    end
    return MPEM2(A)
end

# compute bᵢⱼᵗ(xᵢᵗ,xⱼᵗ) from μᵢⱼ, μⱼᵢ, ψᵢⱼ
# also return normalization zᵢⱼ
function pair_belief(Aᵢⱼ::MPEM2, Aⱼᵢ::MPEM2, ψᵢⱼ)
    A = pair_belief_as_mpem(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    l = accumulate_L(A); r = accumulate_R(A)
    z = only(l[end])
    marginals(A; l, r), z
end