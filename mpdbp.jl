include("bp.jl")
include("dbp_factor.jl")

import Graphs: nv, ne, edges, vertices
import IndexedGraphs: IndexedBiDiGraph, IndexedGraph, inedges, outedges, src, 
                       dst, idx
import UnPack: @unpack
import ProgressMeter: ProgressUnknown, next!
import Random: shuffle!
import Base.Threads: @threads
import SparseArrays: rowvals, nonzeros, nzrange

struct MPdBP{q,T,F<:Real,U<:dBP_Factor}
    g  :: IndexedBiDiGraph{Int}          # graph
    w  :: Vector{Vector{U}}              # factors, one per variable
    ϕ  :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    ψ  :: Vector{Vector{Matrix{F}}}      # edge-dependent factors
    p⁰ :: Vector{Vector{F}}              # prior at time zero
    μ  :: Vector{MPEM2{q,T,F}}           # messages, two per edge
    
    function MPdBP(g::IndexedBiDiGraph{Int}, w::Vector{Vector{U}}, 
            ϕ::Vector{Vector{Vector{F}}}, ψ::Vector{Vector{Matrix{F}}},
            p⁰::Vector{Vector{F}}, 
            μ::Vector{MPEM2{q,T,F}}) where {q,T,F<:Real,U<:dBP_Factor}
    
        @assert length(w) == length(ϕ) == nv(g) "$(length(w)), $(length(ϕ)), $(nv(g))"
        @assert length(ψ) == ne(g)
        @assert all( length(wᵢ) == T for wᵢ in w )
        @assert all( length(ϕ[i][t]) == q for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( size(ψ[ij][t]) == (q,q) for ij in eachindex(ψ) for t in eachindex(ψ[ij]) )
        @assert check_ψs(ψ, g)
        @assert all( length(pᵢ⁰) == q for pᵢ⁰ in p⁰ )
        @assert all( length(ϕᵢ) == T for ϕᵢ in ϕ )
        @assert length(μ) == ne(g)
        return new{q,T,F,U}(g, w, ϕ, ψ, p⁰, μ)
    end
end

getT(bp::MPdBP{q,T,F,U}) where {q,T,F,U} = T
getq(bp::MPdBP{q,T,F,U}) where {q,T,F,U} = q
getN(bp::MPdBP) = nv(bp.g)

# check that factors on edge i→j is the same as the one on j→i
function check_ψs(ψ::Vector{<:Vector{<:Matrix{<:Real}}}, g::IndexedBiDiGraph)
    X = g.X
    N = nv(g)
    rows = rowvals(X)
    vals = nonzeros(X)
    for j in 1:N
        for k in nzrange(X, j)
            i = rows[k]
            if i < j
                ji = k          # idx of edge i→j
                ij = vals[k]    # idx of edge j→i
                check = map(zip(ψ[ij], ψ[ji])) do (ψᵢⱼᵗ, ψⱼᵢᵗ)
                    ψᵢⱼᵗ == ψⱼᵢᵗ'
                end
                all(check) || return false
            end
        end
    end
    return true
end

# return a vector ψ ready for MPdBP starting from observations of the type
#  (i, j, t, ψᵢⱼᵗ)
function pair_observations_directed(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedBiDiGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do e
        # i, j, ij = e
        i = src(e); j = dst(e); ij = idx(e)
        map(1:T) do t
            id_ij = findall(obs->obs[1:3]==(i,j,t), O)
            id_ji = findall(obs->obs[1:3]==(j,i,t), O)
            if !isempty(id_ij)
                cnt += 1
                only(O[id_ij])[4]
            elseif !isempty(id_ji)
                cnt += 1
                only(O[id_ji])[4] |> permutedims
            else
                ones(q, q)
            end
        end
    end
    @assert cnt == 2*length(O)
    ψ
end

function nondirected_to_directed(ψ, g::IndexedGraph)
    
end

function pair_observations_nondirected(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do (i, j, ij)
        map(1:T) do t
            id = findall(obs->(obs[1:3]==(i,j,t) || obs[1:3]==(j,i,t)), O)
            if !isempty(id)
                cnt += 1
                only(O[id])[4]
            else
                ones(q, q)
            end
        end
    end
    @assert cnt == length(O)
    ψ
end

function mpdbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:dBP_Factor}}, 
        q::Int, T::Int; d::Int=1, bondsizes=[1; fill(d, T); 1],
        ϕ = [[ones(q) for t in 1:T] for _ in vertices(g)],
        ψ = [[ones(q,q) for t in 1:T] for _ in edges(g)],
        p⁰ = [ones(q) for i in 1:nv(g)],
        μ = [mpem2(q, T; d, bondsizes) for e in edges(g)])
    return MPdBP(g, w, ϕ, ψ, p⁰, μ)
end

function onebpiter!(bp::MPdBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    for (j_ind, e_out) in enumerate(eout)
        B = f_bp(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        normalize_eachmatrix!(μ[idx(e_out)])
    end
    return nothing
end

struct CB_BP{TP<:ProgressUnknown}
    prog :: TP
    mag :: Vector{Vector{Vector{Float64}}}
    Δs :: Vector{Float64}
    function CB_BP(bp::MPdBP{q,T,F,U}) where {q,T,F,U}
        @assert q == 2
        prog = ProgressUnknown(desc="Running MPdBP: iter")
        TP = typeof(prog)
        mag = [magnetizations(bp)] 
        Δs = zeros(0)
        new{TP}(prog, mag, Δs)
    end
end

function (cb::CB_BP)(bp::MPdBP, it::Integer)
    mag_new = magnetizations(bp)
    mag_old = cb.mag[end]
    Δ = sum(sum(abs, mn .- mo) for (mn,mo) in zip(mag_new,mag_old))
    push!(cb.Δs, Δ)
    next!(cb.prog, showvalues=[(:Δ,Δ)])
    push!(cb.mag, mag_new)
    return Δ
end

function iterate!(bp::MPdBP; maxiter=5, svd_trunc::SVDTrunc=TruncThresh(1e-6),
        cb=CB_BP(bp), tol=1e-10,
        nodes = collect(vertices(bp.g)))
    for it in 1:maxiter
        @threads for i in nodes
            onebpiter!(bp, i; svd_trunc)
        end
        Δ = cb(bp, it)
        Δ < tol && return it, cb
        shuffle!(nodes)
    end
    return maxiter, cb
end

function belief_slow(bp::MPdBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, p⁰, μ = bp
    A = μ[inedges(g,i).|>idx]
    B = f_bp(A, p⁰[i], w[i], ϕ[i])
    C = mpem2(B)
    sweep_RtoL!(C; svd_trunc)
    return firstvar_marginals(C)
end

function beliefs_slow(bp::MPdBP; kw...)
    [belief_slow(bp, i; kw...) for i in vertices(bp.g)]
end

function magnetizations_slow(bp::MPdBP{q,T,F,U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F,U}
    @assert q == 2
    map(vertices(bp.g)) do i
        bᵢ = belief(bp, i; svd_trunc)
        reduce.(-, bᵢ)
    end
end

function magnetizations(bp::MPdBP{q,T,F,U}) where {q,T,F,U}
    @assert q == 2
    map(beliefs(bp)) do bᵢ
        reduce.(-, bᵢ)
    end
end

# compute joint beliefs for all pairs of neighbors
function pair_beliefs(bp::MPdBP{q,T,F,U}) where {q,T,F,U}
    b = [[zeros(q,q) for _ in 0:T] for _ in 1:(ne(bp.g))]
    X = bp.g.X
    N = nv(bp.g)
    rows = rowvals(X)
    vals = nonzeros(X)
    for j in 1:N
        for k in nzrange(X, j)
            i = rows[k]
            if i < j
                ji = k          # idx of message i→j
                ij = vals[k]    # idx of message j→i
                μᵢⱼ = bp.μ[ij]; μⱼᵢ = bp.μ[ji]
                b[ij] .= pair_belief(μᵢⱼ, μⱼᵢ)
                b[ji] .= [bij' for bij in b[ij]]
            end
        end
    end
    b
end

function beliefs(bp::MPdBP; bij = pair_beliefs(bp))
    b = map(vertices(bp.g)) do i 
        ij = idx(first(outedges(bp.g, i)))
        bb = bij[ij]
        map(bb) do bᵢⱼᵗ
            bᵢᵗ = vec(sum(bᵢⱼᵗ, dims=2))
        end
    end
    b
end
