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

seed = 20
rng = MersenneTwister(seed)

T = 10
N = 30
c = 2.5
gg = erdos_renyi(N, c/N; seed)
g = IndexedGraph(gg)

β = 0.5
J = 1.0
h = 0.2
m⁰ = 1.0
K = 30

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

ising = Ising(g, fill(J,ne(g)), fill(h, nv(g)), β)
bp = Glauber(ising, T; ϕ) |> mpbp

g_ = IndexedBiDiGraph(gg)
w_fourier = [fill(GlauberFactor(fill(J,length(inedges(g,i))), h, β), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g_, w_fourier, fill(2, nv(g)), T; ϕ)

matrix_sizes = [5, 10, 35]
maxiters = [20, 20, 15]
iters = zeros(Int, length(maxiters))
tol = 1e-12
for i in eachindex(maxiters)
    iters[i], _ = iterate!(bp; maxiter=maxiters[i], svd_trunc=TruncBond(matrix_sizes[i]), tol)
end

# iters_fourier = zeros(Int, length(maxiters))
# for i in eachindex(maxiters)
#     iters_fourier[i], cb_fourier = iterate_fourier!(bp_fourier, K, maxiter=maxiters[i], σ=1/100; svd_trunc=TruncBond(matrix_sizes[i]), tol)
# end

nsamples = 10^6
sms = SoftMarginSampler(bp)
sample!(sms, nsamples)

potts2spin(x, i; q=2) = (x-1)/(q-1)*2 - 1


using JLD2

m = means(potts2spin, bp)
m_fourier = real.(means(potts2spin, bp_fourier))
traj_mc = [[vec(potts2spin.(X[i,:])) for X in sms.X] for i in 1:N]
m_mc = [mean(x) for x in traj_mc]
σ_mc = [std(x)/sqrt(nsamples) for x in traj_mc]
jldsave("results/comparison_beta0,5_J1_h0,2_er$(N)_bis.jld2"; m, m_fourier, m_mc, σ_mc)