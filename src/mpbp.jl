struct MPBP{G<:AbstractIndexedDiGraph, F<:Real, V<:AbstractVector{<:BPFactor}}
    g  :: G                              # graph
    w  :: Vector{V}              # factors, one per variable
    ϕ  :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    ψ  :: Vector{Vector{Matrix{F}}}      # edge-dependent factors
    μ  :: Vector{MPEM2{F}}               # messages, two per edge
    b  :: Vector{MPEM1{F}}               # beliefs in matrix product form
    
    function MPBP(g::G, w::Vector{V}, 
            ϕ::Vector{Vector{Vector{F}}}, ψ::Vector{Vector{Matrix{F}}},
            μ::Vector{MPEM2{F}}, b::Vector{MPEM1{F}}) where {G<:AbstractIndexedDiGraph,
                                                             F<:Real, V<:AbstractVector{<:BPFactor}}
    
        T = length(w[1]) - 1
        @assert length(w) == length(ϕ) == length(b) == nv(g) "$(length(w)), $(length(ϕ)), $(nv(g))"
        @assert length(ψ) == ne(g)
        @assert all( length(wᵢ) == T + 1 for wᵢ in w )
        @assert all( length(ϕ[i][t]) == nstates(b[i]) for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( size(ψ[k][t]) == (nstates(b[i]),nstates(b[j])) for (i,j,k) in edges(g) for t in eachindex(ψ[k]) )
        @assert check_ψs(ψ, g)
        @assert all( length(ϕᵢ) == T + 1 for ϕᵢ in ϕ )
        @assert all( length(ψᵢ) == T + 1 for ψᵢ in ψ )
        @assert all( getT(μᵢⱼ) == T for μᵢⱼ in μ)
        @assert all( getT(bᵢ) == T for bᵢ in b )
        @assert length(μ) == ne(g)
        normalize!.(μ)
        # normalize observations at time zero because they play the role of the prior
        for i in vertices(g)
            ϕ[i][begin] ./= sum(ϕ[i][begin])
        end
        return new{G,F,V}(g, w, ϕ, ψ, μ, b)
    end
end

getT(bp::MPBP) = getT(bp.b[1])
getN(bp::MPBP) = nv(bp.g)
nstates(bp::MPBP, i) = nstates(bp.b[i])

# check that observation on edge i→j is the same as the one on j→i
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
                for (ψᵢⱼᵗ, ψⱼᵢᵗ) in zip(ψ[ij], ψ[ji])
                    ψᵢⱼᵗ == ψⱼᵢᵗ' || return false
                end
            end
        end
    end
    return true
end

function mpbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}}, q::AbstractVector{Int},
        T::Int; d::Int=1, bondsizes=[1; fill(d, T); 1],
        ϕ = [[ones(q[i]) for t in 0:T] for i in vertices(g)],
        ψ = [[ones(q[i],q[j]) for t in 0:T] for (i,j) in edges(g)],
        μ = [mpem2(q[i],q[j], T; d, bondsizes) for (i,j) in edges(g)],
        b = [mpem1(q[i], T; d, bondsizes) for i in vertices(g)])
    return MPBP(g, w, ϕ, ψ, μ, b)
end

function reset_messages!(bp::MPBP)
    for A in bp.μ
        for Aᵗ in A
            Aᵗ .= 1
        end
        normalize!(A)
    end
    nothing
end

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:BPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    logzᵢ = 0.0
    for (j_ind, e_out) in enumerate(eout)
        B, logzᵢ₂ⱼ = f_bp(A, w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        logzᵢ₂ⱼ += normalize!(μ[idx(e_out)])
        logzᵢ += logzᵢ₂ⱼ
    end
    dᵢ = length(ein)
    bp.b[i] = onebpiter_dummy_neighbor(bp, i; svd_trunc) |> marginalize
    return (1 / dᵢ) * logzᵢ
end

function onebpiter!(bp::MPBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc)
end


function onebpiter_dummy_neighbor(bp::MPBP, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, ψ, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    B, _ = f_bp_dummy_neighbor(A, w[i], ϕ[i], ψ[eout.|>idx]; svd_trunc)
    C = mpem2(B)
    A = sweep_RtoL!(C; svd_trunc)
end

# A callback to print info and save stuff during the iterations 
struct CB_BP{TP<:ProgressUnknown}
    prog :: TP
    m    :: Vector{Vector{Vector{Float64}}} 
    Δs   :: Vector{Float64}

    function CB_BP(bp::MPBP; showprogress::Bool=true)
        dt = showprogress ? 0.1 : Inf
        prog = ProgressUnknown(desc="Running MPBP: iter", dt=dt)
        TP = typeof(prog)

        ## warning :: FIXME and also below
        m = [[marginal_to_expectation.(marginals(bp.b[i]), eltype(bp.w[i]))  for i in eachindex(bp.b)]]
        Δs = zeros(0)
        new{TP}(prog, m, Δs)
    end
end

function (cb::CB_BP)(bp::MPBP, it::Integer)
    marg_new = [marginal_to_expectation.(marginals(bp.b[i]), eltype(bp.w[i])) for i in eachindex(bp.b)]
    marg_old = cb.m[end]
    if isempty(marg_new)
        Δ = NaN
    else
        Δ = mean(mean(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    end
    push!(cb.Δs, Δ)
    push!(cb.m, marg_new)
    next!(cb.prog, showvalues=[(:Δ,Δ)])
    flush(stdout)
    return Δ
end

function iterate!(bp::MPBP; maxiter::Integer=5, 
        svd_trunc::SVDTrunc=TruncThresh(1e-6),
        showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, 
        nodes = collect(vertices(bp.g)), shuffle::Bool=true)
    for it in 1:maxiter
        for i in nodes
            onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc)
        end
        Δ = cb(bp, it)
        Δ < tol && return it, cb
        shuffle && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return maxiter, cb
end

# compute joint beliefs for all pairs of neighbors
# return also logzᵢⱼ contributions to logzᵢ
function pair_beliefs(bp::MPBP{G,F}) where {G,F}
    b = [[zeros(nstates(bp,i),nstates(bp,j)) for _ in 0:getT(bp)] for (i,j) in edges(bp.g)]
    logz = zeros(nv(bp.g))
    X = bp.g.X
    N = nv(bp.g)
    rows = rowvals(X)
    vals = nonzeros(X)
    for j in 1:N
        dⱼ = length(nzrange(X, j))
        for k in nzrange(X, j)
            i = rows[k]
            ji = k          # idx of message i→j
            ij = vals[k]    # idx of message j→i
            μᵢⱼ = bp.μ[ij]; μⱼᵢ = bp.μ[ji]
            bᵢⱼ, zᵢⱼ = pair_belief(μᵢⱼ, μⱼᵢ, bp.ψ[ij])
            logz[j] += (1/dⱼ- 1/2) * log(zᵢⱼ)
            b[ij] .= bᵢⱼ
        end
    end
    b, logz
end

beliefs(bp::MPBP{G,F}) where {G,F} = marginals.(bp.b)

beliefs_tu(bp::MPBP{G,F}) where {G,F} = marginals_tu.(bp.b)


# function beliefs(bp::MPBP{G,F}; bij = pair_beliefs(bp)[1]) where {G,F}
#     b = map(vertices(bp.g)) do i 
#         ij = idx(first(outedges(bp.g, i)))
#         bb = bij[ij]
#         map(bb) do bᵢⱼᵗ
#             bᵢᵗ = vec(sum(bᵢⱼᵗ, dims=2))
#         end
#     end
#     b
# end


function marginal_to_expectation(p::Matrix{<:Real}, U::Type{<:BPFactor})
    μ = 0.0
    for xi in axes(p,1) , xj in axes(p, 2)
        μ += idx_to_value(xi, U) * idx_to_value(xj, U) * p[xi, xj]
    end
    μ
end

function marginal_to_expectation(p::Vector{<:Real}, U::Type{<:BPFactor})
    μ = 0.0
    for xi in eachindex(p)
        μ += idx_to_value(xi, U) * p[xi]
    end
    μ
end


# beliefs_tu(bp::MPBP{G,F}) where {G,F} = marginals_tu.(bp.b)


function autocorrelation(b_tu::Matrix{Matrix{F}}, 
        U::Type{<:BPFactor}) where {F<:Real}
    r = zeros(size(b_tu)...)
    for t in axes(b_tu, 1), u in axes(b_tu, 2)
        r[t,u] = marginal_to_expectation(b_tu[t,u], U)
    end
    r
end

function autocorrelations(bp::MPBP{G,F};
        b_tu = beliefs_tu(bp)) where {G,F}
    autocorrelations(b_tu, eltype(bp.w[1])) #FIXME
end

function autocorrelations(b_tu::Vector{Matrix{Matrix{F}}},
        U::Type{<:BPFactor}) where {F<:Real}
    [autocorrelation(bi_tu, U) for bi_tu in b_tu]
end

function covariance(r::Matrix{F}, μ::Vector{F}) where {F<:Real}
    c = zero(r)
    for u in axes(r,2), t in 1:u-1
        c[t, u] = r[t, u] - μ[t]*μ[u]
    end
    c
end

# r: autocorrelations, μ: expectations of marginals
function _autocovariances(r::Vector{Matrix{F}}, μ::Vector{Vector{F}}) where {F<:Real}
    map(eachindex(r)) do i
        covariance(r[i], μ[i])
    end
end

function autocovariances(bp::MPBP{G,F}; 
        r = autocorrelations(bp), m = beliefs(bp)) where {G,F}
    μ = [marginal_to_expectation.(mᵢ, eltype(wi)) for (wi,mᵢ) in zip(bp.w,m)] 
    _autocovariances(r, μ)
end

function bethe_free_energy(bp::MPBP; svd_trunc=TruncThresh(1e-4))
    fa = zeros(getN(bp))
    for i in eachindex(fa)
        logzi = onebpiter!(bp, i; svd_trunc)
        fa[i] -= logzi
    end
    _, logz_edges = pair_beliefs(bp)
    fa .-= logz_edges
    sum(fa)
end

# compute log of posterior probability for a trajectory `x`
function logprob(bp::MPBP, x::Matrix{<:Integer})
    @unpack g, w, ϕ, ψ, μ = bp
    N = nv(bp.g); T = getT(bp)
    @assert size(x) == (N , T + 1)
    logp = 0.0

    for i in 1:N
        logp += log(ϕ[i][1][x[i,1]])
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            @views logp += log( w[i][t](x[i, t+1], x[∂i, t], x[i, t]) )
            logp += log( ϕ[i][t+1][x[i, t+1]] )
        end
    end
    for t in 1:T+1
        for (i, j, ij) in edges(bp.g)
            logp += 1/2 * log( ψ[ij][t][x[i,t], x[j,t]] )
        end
    end
    return logp
end


# return a vector ψ ready for MPBP starting from observations of the type
#  (i, j, t, ψᵢⱼᵗ)
function pair_observations_directed(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedBiDiGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do (i, j, ij)
        map(0:T) do t
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

function pair_observations_nondirected(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do (i, j, ij)
        map(0:T) do t
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

function pair_obs_undirected_to_directed(ψ_undirected::Vector{<:F}, 
        g::IndexedGraph) where {F<:Vector{<:Matrix}}
    ψ_directed = F[]
    sizehint!(ψ_directed, 2*length(ψ_directed)) 
    A = g.A
    vals = nonzeros(A)
    rows = rowvals(A)

    for j in 1:nv(g)
        for k in nzrange(A, j)
            i = rows[k]
            ij = vals[k]
            if i < j
                push!(ψ_directed, ψ_undirected[ij])
            else
                push!(ψ_directed, [permutedims(ψᵢⱼᵗ) for ψᵢⱼᵗ in ψ_undirected[ij]])
            end
        end
    end

    ψ_directed
end


onebpiter