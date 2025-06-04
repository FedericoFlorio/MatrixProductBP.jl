using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Distributions
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using Plots

# T = 10
# c = 4.0
# m⁰ = -0.6

# β = 1.0
# h = 0.0

# popsize = 10^2
# svd_trunc = TruncBond(5)
# K = 20
# σ = 1/30

# ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
# ψ_neutral = [ones(2,2) for t in 0:T]

# prob_degree = Poisson(c)
# prob_J = Uniform(0.0, 1.0)
# prob_h = Dirac(0.0)

# μ_pop = map(1:popsize) do p
#     μ = rand_mpem2(ComplexF64, 2, 2, T)
#     normalize!(μ)
#     μ
# end
# bs = Vector{Vector{Float64}}[]
# bs2var =  Matrix{Matrix{Float64}}[]


# iterate_fourier_popdyn!(μ_pop, popsize, bs, bs2var, prob_degree, prob_J, prob_h, K, β, ϕᵢ, T; maxiter=1000, svd_trunc, tol=1e-10, σ)

# ns = 500
# range = length(btus)+1-min(ns, length(btus)):length(btus)
# ms = [expectation.(potts2spin, b) for b in bs[range]]
# m_avg = mean(ms)
# m_std = std(ms) ./ sqrt(length(ms))

function glauber_factors_(ising::Ising, T::Integer)
    β = ising.β
    map(1:nv(ising.g)) do i
        ei = inedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = GlauberFactor(J, h, β)
        fill(wᵢᵗ, T + 1)
    end
end
function mpbp_(gl::Glauber{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors_(gl.ising, T)
    ϕ = gl.ϕ
    ψ = pair_obs_undirected_to_directed(gl.ψ, gl.ising.g)
    return mpbp(g, w, fill(2, nv(g)), T; ϕ, ψ, kw...)
end

seed = 1
N = 5*10^3
g = erdos_renyi(N, c/N; seed)
ising = Ising(IndexedGraph(g); J=rand(ne(g)), h=fill(h,N), β)
bp = mpbp_(Glauber(ising, T); ϕ = fill(ϕᵢ, N))
sms = SoftMarginSampler(bp)
sample!(sms, 5*10^3)
traj_mc = [vec(potts2spin.(mean(X, dims=1))) for X in sms.X]

m_mc = mean(traj_mc)
σ_mc = std(traj_mc) ./ sqrt(N)

using JLD2
jldsave("results/popdyn_infinite_er.jld2"; bs, bs2var, m_mc, σ_mc)