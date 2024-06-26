struct SIS_heterogeneous{T, N, F<:Real}
    g  :: IndexedGraph
    λ  :: SparseMatrixCSC{F}         # infection probabilities
    ρ  :: Vector{F}                  # recovery probabilities
    α  :: Vector{F}                  # auto-infection probabilities
    ϕ  :: Vector{Vector{Vector{F}}}  # site observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge observations
    function SIS_heterogeneous(g::IndexedGraph, λ::SparseMatrixCSC{F,Int64}, ρ::Vector{F},
        α::Vector{F},
        ϕ::Vector{Vector{Vector{F}}},
        ψ::Vector{Vector{Matrix{F}}}) where {F<:Real}
        @assert size(λ)[1] == size(λ)[2] == nv(g)
        @assert length(ρ) == nv(g)
        @assert all(0 ≤ λᵢⱼ ≤ 1 for λᵢⱼ in !iszero(λ))
        @assert all(0 ≤ ρᵢ ≤ 1 for ρᵢ in ρ)
        @assert all(0 ≤ αᵢ ≤ 1 for αᵢ in α)
        N = nv(g)
        @assert length(ϕ) == N
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T + 1 for ϕᵢ in ϕ)
        @assert length(ψ) == 2*ne(g)
        @assert all(length(ψᵢⱼ) == T + 1 for ψᵢⱼ in ψ)
        new{T,N,F}(g, λ, ρ, α, ϕ, ψ)
    end
end

function SIS_heterogeneous(g::IndexedGraph{Int}, λ::SparseMatrixCSC{F,Int64}, ρ::Vector{F}, T::Int;
        α = zeros(size(λ,1)),
        ψ = [[ones(2,2) for t in 0:T] for _ in 1:2*ne(g)],
        γ = 0.5,
        ϕ = [[t == 0 ? (length(γ)==1 ? [1-γ, γ] : [1-γ[i],γ[i]]) : ones(2) for t in 0:T] for i in vertices(g)]) where {F<:Real}
        
    return SIS_heterogeneous(g, λ, ρ, α, ϕ, ψ)
end

function SIS_heterogeneous(λ::SparseMatrixCSC{F,Int64}, ρ::Vector{F}, T::Int; γ=0.5, α::Vector{F}=zeros(size(λ,1))) where {F<:Real}
    A = ones(Int,size(λ)[1],size(λ)[2]) - iszero.(λ)
    g = IndexedGraph(A+A')
    
    return SIS_heterogeneous(g, λ, ρ, T; γ, α)
end

# WARNING! The λs are all bound together
function sis_heterogeneous_factors(sis::SIS_heterogeneous{T,N,F}) where {T,N,F}
    [[SIS_heterogeneousFactor(nonzeros(sis.λ)[nzrange(sis.λ,i)], sis.ρ[i]; α=sis.α[i]) for _ in 1:T+1] for i in vertices(sis.g)]
end