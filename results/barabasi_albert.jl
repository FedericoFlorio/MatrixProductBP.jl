using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using Plots

seed = 1
rng = MersenneTwister(seed)

T = 10
N = 20
gg = barabasi_albert(N, 5, 2; rng, complete=true)
g = IndexedBiDiGraph(gg)

β = 0.7
h = 0.1
m⁰ = 0.1
K = 100
σ = 1/60

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    j>i && continue
    if has_edge(gg,i,j)
        J[i,j] = 2*rand(rng)-1
        J[j,i] = J[i,j]
    end
end

for i in vertices(g)
    print("$(i):\t")
    for j in inedges(g, i)
        print("$j\t")
    end
    println()
end

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

w_fourier = [fill(GlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

matrix_sizes = [5, 10, 15]
maxiters = [10, 0, 10]
tol = 1e-16

iters_fourier = zeros(Int, length(maxiters))
for i in eachindex(maxiters)
    iters_fourier[i], cb_fourier = iterate_fourier!(bp_fourier, K; maxiter=maxiters[i], σ, svd_trunc=TruncBond(matrix_sizes[i]), tol)
end

nsamples = 10^6
sms = SoftMarginSampler(bp_fourier)
sample!(sms, nsamples)

m_fourier = real.(means(potts2spin, bp_fourier))
traj_mc = [[vec(potts2spin.(X[i,:])) for X in sms.X] for i in 1:N]
m_mc = [mean(x) for x in traj_mc]
σ_mc = [std(x)/sqrt(nsamples) for x in traj_mc]

using JLD2
jldsave("results/random_coupling_beta0,7_h0,1_ba$(N).jld2"; m_fourier, m_mc, σ_mc)