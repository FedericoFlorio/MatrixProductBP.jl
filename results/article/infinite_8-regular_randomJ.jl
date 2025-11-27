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

popsize = 10^2
bonddims = [5, 10, 15]
maxiters = [20, 50, 120]
K = 60
σ = 1/100

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]

prob_degree = Dirac(k)
prob_J = Uniform(-1.0, 1.0)
prob_h = Dirac(0.0)

μ_pop = map(1:popsize) do p
    μ = rand_mpem2(ComplexF64, 2, 2, T)
    normalize!(μ)
    μ
end |> AtomicVector
bs = Vector{Vector{Float64}}[] |> AtomicVector
bs2times =  Matrix{Matrix{Float64}}[] |> AtomicVector
bs2vars =  Tuple{Matrix{Array{Float64,4 }}, Float64, Int64}[] |> AtomicVector

for ind in eachindex(bonddims)
    d = bonddims[ind]
    maxiter = maxiters[ind]

    iterate_fourier_popdyn!(μ_pop, popsize, bs, bs2times, bs2vars, prob_degree, prob_J, prob_h, K, β, ϕᵢ, T; maxiter, svd_trunc=TruncBond(d), tol=1e-10, σ, parallel=10)
end


Nmc = 5*10^3
g = random_regular_graph(Nmc, k) |> IndexedBiDiGraph
J = [rand() for _ in edges(g)] .*2 .- 1.0
h = 0.0
ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
w_mc = [fill(GenericGlauberFactor(Float64[J[ed|>idx] for ed in inedges(g,i)].*β, h*β), T+1) for i in vertices(g)]
bp_mc = mpbp(Float64, g, w_mc, fill(2, nv(g)), T; ϕ = fill(ϕᵢ, Nmc))
sms = SoftMarginSampler(bp_mc)

X = zeros(Int, Nmc, T+1)
autocorrs_mc = [zeros(T+1) for _ in 1:Nmc]
means_mc = [zeros(T+1) for _ in 1:Nmc]
energy_mc = zeros(T)

nsamples = 10^4
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
jldsave("results/article/popdyn_infinite_8-reg_disordered_posneg_beta0,3_dmax$(bonddims[end])_K$(K).jld2"; bs, bs2times, bs2vars)
jldsave("results/article/monte_carlo_infinite_8-reg_disordered_posneg_beta0,3_Nmc$(Nmc)_nsamp$(nsamples).jld2"; m_mc, autocorr_mc, energy_mc)