using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Tullio, ProgressMeter
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact

seed = 1
rng = MersenneTwister(seed)

T = 10
N = 200

kmax = 20   # maximum degree
γ = 0.6     # power-law exponent
ω = Int[round(kmax*i^-γ) for i in 1:N]
gg = random_configuration_model(N, ω; rng, check_graphical=true)
g = IndexedBiDiGraph(gg)
for i in vertices(g)
    println("$i \t\t k = $(degree(g,i))")
end

β = 0.7
h = 0.0
m⁰ = 0.6
K = 100
σ = 1/100
P = 2.0
p = 0.1

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    # j>i && continue
    if has_edge(gg,i,j)
        J[i,j] = 2*rand(rng)-1
        # J[i,j] = rand(rng)
        # J[i,j] = 1.0
        J[j,i] = J[i,j]
    end
end

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

w_fourier = [fill(FourierGlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β; K, σ, P, p), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

bondsizes = [8]
maxiters = [20]
tol = 1e-10

iters_fourier = zeros(Int, length(maxiters))
for i in eachindex(maxiters)
    iters_fourier[i], cb_fourier = iterate!(bp_fourier; maxiter=maxiters[i], svd_trunc=TruncBond(bondsizes[i]), tol)
end

m_fourier = real.(means(potts2spin, bp_fourier))

probs(x,t) = x[t,t+1]
energy_fourier = zeros(N,N,T)
for edout in edges(g)
    i,j = src(edout), dst(edout)
    edin = get_edge(g,j,i)

    μᵢⱼ = bp_fourier.μ[edout|>idx]
    μⱼᵢ = bp_fourier.μ[edin|>idx]
    bᵢⱼ = MatrixProductBP.pair_belief_as_mpem(μᵢⱼ, μⱼᵢ, ψ_neutral)
    Jⱼᵢ = J[j,i]
    pᵢⱼ = [real.(m) for m in twovar_marginals(bᵢⱼ)]
    p_ = [(@tullio _[xᵢᵗ⁺¹, xⱼᵗ] := probs(pᵢⱼ, $t)[xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹,xⱼᵗ⁺¹]) for t in 1:T]

    energy_fourier[i,j,:] .= Jⱼᵢ .* expectation.(potts2spin, p_)
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
jldsave("article/configuration_$(N)_0,6_beta0,7_h0_randomJ.jld2"; m_fourier, energy_fourier, J)
# jldsave("article/monte_carlo_configuration_$(N)_0,6_randomJ_beta0,7_h0_nsamp$(nsamples).jld2"; m_mc, energy_mc)