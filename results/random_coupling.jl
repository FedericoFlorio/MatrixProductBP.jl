using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using Plots
include("../fourier_tensor_train.jl")
include("../bp_fourier.jl")

seed = 5
rng = MersenneTwister(seed)

T = 50
N = 10
c = 2.0
gg = erdos_renyi(N, c/N; seed)
g = IndexedBiDiGraph(gg)

β = 1.0
h = 0.0
m⁰ = 0.2
K = 30

svd_trunc=TruncBond(20)

display(connected_components(gg))

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    j>i && continue
    if has_edge(gg,i,j)
        J[i,j] = 2*rand(rng)-1
        J[j,i] = J[i,j]
    end
end

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

w_fourier = [fill(GlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

iters, cb_fourier = iterate_fourier!(bp_fourier,K, maxiter=50, σ=1/100; svd_trunc, tol=1e-10)

potts2spin(x, i; q=2) = (x-1)/(q-1)*2 - 1

m_fourier = real.(means(potts2spin, bp_fourier))

nsamples = 10^6
sms = SoftMarginSampler(bp_fourier)
sample!(sms, nsamples)
traj_mc = [[vec(potts2spin.(X[i,:])) for X in sms.X] for i in 1:N]
m_mc = [mean(x) for x in traj_mc]
σ_mc = [std(x)./sqrt(nsamples) for x in traj_mc]

using JLD2
jldsave("results/random_coupling_er$(N)_bis.jld2"; m_fourier, m_mc, σ_mc)