using Pkg
Pkg.activate("C:/Users/fefif/Desktop/PhD/Progetti/MatrixProductBP.jl")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, Plots
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
include("C:/Users/fefif/Desktop/PhD/Progetti/Utilities/roc.jl");

nsnaps = 100
separation = 256
T = nsnaps * separation
N = 10
seed = 4

c = 2
gg = erdos_renyi(N, c/N; seed)
g = IndexedGraph(gg)

λ_unif = 0.02
ρ_unif = 0.02
λ = zeros(N,N)
for i in CartesianIndices(λ)
    if !iszero(g.A[i])
        # λ[i] = rand()
        λ[i] = λ_unif
    end
end
λ = sparse(λ)
# ρ = rand(N)
ρ = fill(ρ_unif,N)
# γ = [i==4 ? 1.0 : 0.0 for i in 1:N]
γ = 0.2
α = fill(1e-4,N)

# T = 7
# N = 2
# seed = 6

# A = [0 1; 1 0]
# g = IndexedGraph(A)

# λ_unif = 0.7
# ρ_unif = 0.6
# λ = sparse(λ_unif .* A)
# # λ = sparse([0 1e-12; λ_unif 0])
# ρ = fill(ρ_unif, N)
# γ = 0.5

sis = SIS_heterogeneous(λ, ρ, T; γ, α);
bp_obs = mpbp(sis);

obs_times = collect(range(0, step=separation, length=nsnaps))
nobs = floor(Int, N * length(obs_times) * 1.0)
obs_fraction = nobs / N
seed = 3
rng = MersenneTwister(seed)
X, observed = draw_node_observations!(bp_obs, nsnaps, times = obs_times .+ 1, softinf=Inf; rng);

λinit = 0.5
ρinit = 0.5

A_complete = ones(N,N) - I
g_complete = IndexedGraph(A_complete)
λ_complete = sparse(λinit.*A_complete)
ρ_complete = fill(ρinit, N)

sis_inf = SIS_heterogeneous(g_complete, λ_complete, ρ_complete, T; γ, ϕ=deepcopy(bp_obs.ϕ))
bp_inf = mpbp(sis_inf);

svd_trunc = TruncBond(5)
maxiter = 40

nodes = vertices(bp_obs.g)

λder = [zeros(length(nodes)-1) for n in nodes]
ρder = zeros(length(nodes))
params = []

iters, cb = inference_parameters!(bp_inf, method=31, maxiter=maxiter, λstep=0.01, ρstep=0.01);