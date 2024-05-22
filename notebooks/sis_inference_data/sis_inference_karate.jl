using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, PyPlot, DelimitedFiles
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
include("/home/fedflorio/master_thesis/Utilities/roc.jl")

A = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/karate.txt", Bool)
g = IndexedGraph(A)

nsnaps = 100
separation = 2
T = nsnaps * separation
N = nv(g)
seed = 4

λ_unif = 0.4
ρ_unif = 0.3
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
γ = [i==1 ? 1.0 : 0.0 for i in 1:N]

sis = SIS_heterogeneous(λ, ρ, T; γ)
bp_obs = mpbp(sis)

obs_times = collect(range(separation, step=separation, length=nsnaps))
nobs = floor(Int, N * length(obs_times) * 1.0)
obs_fraction = nobs / N
rng = MersenneTwister(seed)
X, observed = draw_node_observations!(bp_obs, nobs, times = obs_times .+ 1, softinf=Inf; rng)

λinit = 0.1
ρinit = 0.1

A_complete = ones(N,N) - I
g_complete = IndexedGraph(A_complete)
λ_complete = sparse(λinit.*A_complete)
ρ_complete = fill(ρinit, N)

sis_inf = SIS_heterogeneous(g_complete, λ_complete, ρ_complete, T; γ, ϕ=deepcopy(bp_obs.ϕ))
bp_inf = mpbp(sis_inf)


svd_trunc = TruncBond(5)
maxiter = 40

iters, cb = inference_parameters!(bp_inf, method=31, maxiter=maxiter, λstep=0.01, ρstep=0.01, logpriorder=(x)->0.0)

data = cb.data[end]
xplot, yplot, auc = roccurve(cb.data[end].λ, λ)
jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/karate_snaps_step2_nobs100.jld2"; data, λ)

@show auc