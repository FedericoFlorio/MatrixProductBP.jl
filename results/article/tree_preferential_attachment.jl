using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Tullio, ProgressMeter
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using Plots

seed = 1
rng = MersenneTwister(seed)

T = 10
N = 10
gg = barabasi_albert(N, 2, 1; rng, complete=true)
g = IndexedBiDiGraph(gg)

β = 0.5
h = 0.1
m⁰ = 0.7
K = 100
σ = 1/100

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    # j>i && continue
    if has_edge(gg,i,j)
        J[i,j] = 2*rand(rng)-1
        # J[i,j] = rand(rng)
        # J[i,j] = 1.0
        # J[j,i] = J[i,j]
    end
end

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

w_fourier = [fill(GlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

bondsizes = [15]
maxiters = [20]
tol = 1e-10

iters_fourier = zeros(Int, length(maxiters))
for i in eachindex(maxiters)
    iters_fourier[i], cb_fourier = iterate_fourier!(bp_fourier, K; maxiter=maxiters[i], σ, svd_trunc=TruncBond(bondsizes[i]), tol)
end

m_fourier = real.(means(potts2spin, bp_fourier))

probs(x,t) = x[t,t+1]
energy_fourier = zeros(N,N,T)
for edout in edges(g)
    i,j = src(edout), dst(edout)
    edin = get_edge(g,j,i)

    μᵢⱼ = bp_fourier.μ[edout|>idx]
    μⱼᵢ = bp_fourier.μ[edin|>idx]
    bᵢⱼ = pairbelief(μᵢⱼ, μⱼᵢ)
    Jⱼᵢ = J[j,i]
    pᵢⱼ = [real.(m) for m in twovar_marginals(bᵢⱼ)]
    p = [(@tullio _[xᵢᵗ⁺¹, xⱼᵗ] := probs(pᵢⱼ, $t)[xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹,xⱼᵗ⁺¹]) for t in 1:T]

    energy_fourier[i,j,:] .= Jⱼᵢ .* expectation.(potts2spin, p)
end

Nmc = N
nsamples = 10^6
w_mc = [fill(GenericGlauberFactor(Float64[J[src(ed),i] for ed in inedges(g,i)].*β, h*β), T+1) for i in vertices(g)]
bp_mc = mpbp(Float64, g, w_mc, fill(2, nv(g)), T; ϕ = fill(ϕᵢ, Nmc))
sms = SoftMarginSampler(bp_mc)

X = zeros(Int, Nmc, T+1)
# autocorrs_mc = [zeros(T+1) for _ in 1:Nmc]
m_mc = [zeros(T+1) for _ in 1:Nmc]
energy_mc = [zeros(T) for _ in 1:Nmc, _ in 1:Nmc]

@showprogress for samp in 1:nsamples
    onesample!(X, bp_mc)
    for i in 1:Nmc
        # autocorrs_mc[i] .+= potts2spin.(X[i,:]) .* potts2spin(X[i,end])
        m_mc[i] .+= @views potts2spin.(X[i,:])
    end
    for ed in edges(g)
        j,i = ed.src, ed.dst
        energy_mc[i,j] .+= @views potts2spin.(X[i,2:end]) .* potts2spin.(X[j,1:end-1]) .* J[j,i]
    end
end

# autocorrs_mc ./= nsamples
m_mc ./= nsamples
# autocorrs_mc .-= m_mc .* [x[end] for x in m_mc]
energy_mc ./= nsamples

using JLD2
jldsave("results/article/tree_$(N)_beta0,5_h0,1_randomJ.jld2"; m_fourier, energy_fourier, J)
jldsave("results/article/monte_carlo_tree_$(N)_disordered_posneg_beta0,5_Nmc$(Nmc)_nsamp$(nsamples).jld2"; m_mc, autocorr_mc, energy_mc)