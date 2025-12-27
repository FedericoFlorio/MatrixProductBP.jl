using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using Plots, LaTeXStrings

seed = 1
rng = MersenneTwister(seed)

T = 10
N = 10
gg = barabasi_albert(N, 2, 1; rng, complete=true)
g = IndexedBiDiGraph(gg)

β = 0.5
h = 0.2
m⁰ = 0.7
K = 40
σ = 1/40

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    j>i && continue
    if has_edge(gg,i,j)
        # J[i,j] = 2*rand(rng)-1
        # J[i,j] = rand(rng)
        J[i,j] = 1.0
        J[j,i] = J[i,j]
    end
end

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))

w_fourier = [fill(GlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β), T+1) for i in vertices(g)]
bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

bondsize = 15
maxiter = 10
tol = 1e-12

using Profile
Profile.clear()
@profview iterate_fourier!(bp_fourier, K; maxiter, σ, svd_trunc=TruncBond(bondsize), tol)