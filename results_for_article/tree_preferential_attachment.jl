# using Pkg
# Pkg.activate(".")

# using Revise
# using MatrixProductBP, MatrixProductBP.Models
# using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Tullio, ProgressMeter
# import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
# using TensorTrains: summary_compact

# seed = 1
# rng = MersenneTwister(seed)

# T = 10
# N = 100
# gg = barabasi_albert(N, 2, 1; rng, complete=true)
# g = IndexedBiDiGraph(gg)

# β = 0.5
# h = 0.0
# m⁰ = 0.7
# K = 30
# σ = 1/30
# P = 2.0
# p = 0.1

# J = zeros(nv(g),nv(g))
# for i in axes(J)[1], j in axes(J)[2]
#     # j>i && continue
#     if has_edge(gg,i,j)
#         J[i,j] = 2*rand(rng)-1
#         # J[i,j] = rand(rng)
#         # J[i,j] = 1.0
#         J[j,i] = J[i,j]
#     end
# end

# ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
# ψ_neutral = [ones(2,2) for t in 0:T]
# ϕ = fill(ϕᵢ, nv(g))

# w_fourier = [fill(FourierGlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β; K, σ, P, p), T+1) for i in vertices(g)]
# bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

# bondsizes = 2:4:22
# basesizes = 10:20:90
# tol = 1e-10

# iters_fourier = zeros(Int, length(bondsizes), length(basesizes))
# bp_fourier = Any[0 for i in bondsizes, j in basesizes]
# for i in eachindex(bondsizes), j in eachindex(basesizes)
#     @show bondsizes[i] basesizes[j]
#     σ = 1/basesizes[j]
#     w_fourier = [fill(FourierGlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β; K=basesizes[j], σ, P, p), T+1) for i in vertices(g)]
#     bp_fourier[i,j] = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

#     maxiter = 30
#     iters_fourier[i,j], cb_fourier = iterate!(bp_fourier[i,j]; maxiter, svd_trunc=TruncBond(bondsizes[i]), tol)
# end

# m_fourier = [real.(means(potts2spin, bp_f)) for bp_f in bp_fourier]


probs(x,t) = x[t,t+1]
energy_fourier = [zeros(N,N,T) for d in bondsizes, K in basesizes]
for dK in eachindex(energy_fourier)
    en = energy_fourier[dK]
# for en in energy_fourier
    for edout in edges(g)
        i,j = src(edout), dst(edout)
        edin = get_edge(g,j,i)

        μᵢⱼ = bp_fourier[dK].μ[edout|>idx]
        μⱼᵢ = bp_fourier[dK].μ[edin|>idx]
        bᵢⱼ = MatrixProductBP.pair_belief_as_mpem(μᵢⱼ, μⱼᵢ, ψ_neutral)
        Jⱼᵢ = J[j,i]
        pᵢⱼ = [real.(m) for m in twovar_marginals(bᵢⱼ)]
        p_ = [(@tullio _[xᵢᵗ⁺¹, xⱼᵗ] := probs(pᵢⱼ, $t)[xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹,xⱼᵗ⁺¹]) for t in 1:T]

        en[i,j,:] .= Jⱼᵢ .* expectation.(potts2spin, p_)
    end
end

Nmc = N
nsamples = 10^6
w_mc = [fill(DampedGlauberFactor(Float64[J[src(ed),i] for ed in inedges(g,i)], h, β, p), T+1) for i in vertices(g)]
bp_mc = mpbp(Float64, g, w_mc, fill(2, nv(g)), T; ϕ = fill(ϕᵢ, Nmc))
sms = SoftMarginSampler(bp_mc)

X = zeros(Int, Nmc, T+1)
m_mc = [zeros(T+1) for _ in 1:Nmc]
energy_mc = zeros(T)

@showprogress for samp in 1:nsamples
    onesample!(X, bp_mc)
    for i in 1:Nmc
        m_mc[i] .+= @views potts2spin.(X[i,:])
    end
    for ed in edges(g)
        j,i = ed.src, ed.dst
        energy_mc .+= @views potts2spin.(X[i,2:end]) .* potts2spin.(X[j,1:end-1]) .* J[j,i]
    end
end

m_mc ./= nsamples
energy_mc ./= nsamples

using JLD2
jldsave("tree_$(N)_beta0,5_h0_randomJ_2.jld2"; m_fourier, energy_fourier, J)
jldsave("monte_carlo_tree_$(N)_randomJ_beta0,5_h0_nsamp$(nsamples)_2.jld2"; m_mc, energy_mc)