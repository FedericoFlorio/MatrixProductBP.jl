using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Distributions
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
import ProgressMeter: @showprogress
using TensorTrains: summary_compact

T = 10
k = 8
m⁰ = 0.6
β = 0.3
K = 60
σ = 1/60
P = 2.0
p = 0.3
prob_degree = Dirac(k)

popsize = 10^2
bonddims = [5, 10, 15]
maxiters = [200, 500, 1000]

function prob_w(w; d=dᵢ)
    T = length(w) - 1

    prob_J = Uniform(-1.0, 1.0)
    prob_h = Dirac(0.0)
    Js = rand(prob_J, d)
    h = rand(prob_h)

    fill(FourierGlauberFactor(Js,h,β; K,σ,P,p), T+1)
end

w_init = fill(FourierGlauberFactor([0.0],0.0,1.0; K,σ,P,p), T+1)

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]


μ_pop = map(1:popsize) do p
    μ = rand_mpem2(ComplexF64, 2, 2, T)
    normalize!(μ)
    μ
end |> AtomicVector
bs = Vector{Vector{Float64}}[] |> AtomicVector
bs2times =  Matrix{Matrix{Float64}}[] |> AtomicVector
bs2vars =  Tuple{Matrix{Array{Float64,4 }}, Float64, Int64}[] |> AtomicVector

function stats!((bs, bs2times, bs2vars), wᵢ, μ, μin, b, f)
    push!(bs, [real.(m) for m in marginals(b)])
    push!(bs2times, [real.(m) for m in twovar_marginals(b)])
    b_pairs = map(eachindex(μ, μin)) do j
        μj = μ[j]
        μinj = μin[j]
        MatrixProductBP.pair_belief_as_mpem(μj, μinj, ψ_neutral)
    end
    for (j, b_pair) in enumerate(b_pairs)
        push!(bs2vars, ([real.(m) for m in twovar_marginals(b_pair)], wᵢ[1].J[j], length(wᵢ[1].J)))
    end
end

for ind in eachindex(bonddims)
    d = bonddims[ind]
    maxiter = maxiters[ind]

    iterate_popdyn!(μ_pop, w_init, prob_degree, prob_w, (bs, bs2times, bs2vars); ϕ=ϕᵢ, stats=stats!, T, maxiter, svd_trunc=TruncBond(d))
end


Nmc = 5*10^3
g = random_regular_graph(Nmc, k) |> IndexedBiDiGraph
J = [rand() for _ in edges(g)] .*2 .- 1.0
h = 0.0
ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
w_mc = [fill(DampedGlauberFactor(Float64[J[ed|>idx] for ed in inedges(g,i)], h, β, p), T+1) for i in vertices(g)]
bp_mc = mpbp(Float64, g, w_mc, fill(2, nv(g)), T; ϕ = fill(ϕᵢ, Nmc))
sms = SoftMarginSampler(bp_mc)

X = zeros(Int, Nmc, T+1)
autocorrs_mc = [zeros(T+1) for _ in 1:Nmc]
means_mc = [zeros(T+1) for _ in 1:Nmc]
energy_mc = zeros(T)

nsamples = 10^6
@showprogress for samp in 1:nsamples
    onesample!(X, bp_mc)
    for i in 1:Nmc
        autocorrs_mc[i] .+= potts2spin.(X[i,:]) .* potts2spin(X[i,end])
        means_mc[i] .+= potts2spin.(X[i,:])
    end
    for ed in edges(g)
        j, i = src(ed), dst(ed)
        energy_mc .+= potts2spin.(X[i,2:end]) .* potts2spin.(X[j,1:end-1]) .* J[ed|>idx]
    end
end

autocorrs_mc ./= nsamples
means_mc ./= nsamples
autocorrs_mc .-= means_mc .* [x[end] for x in means_mc]
autocorr_mc = mean([abs.(x) for x in autocorrs_mc])
m_mc = mean(means_mc)
energy_mc ./= nsamples * Nmc * k

using JLD2
jldsave("article/popdyn_infinite_8-reg_disordered_posneg_beta0,3_dmax$(bonddims[end])_K$(K).jld2"; bs, bs2times, bs2vars)
jldsave("article/monte_carlo_infinite_8-reg_disordered_posneg_beta0,3_Nmc$(Nmc)_nsamp$(nsamples).jld2"; m_mc, autocorr_mc, energy_mc)