using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, PyPlot, DelimitedFiles, SpecialFunctions
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using JLD2
include("/home/fedflorio/master_thesis/Utilities/roc.jl")


nsnaps = 100
separation = 12
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

sis = SIS_heterogeneous(λ, ρ, T; γ, α)
bp_obs = mpbp(sis)

obs_times = collect(range(0, step=separation, length=nsnaps))
nobs = floor(Int, N * length(obs_times) * 1.0)
obs_fraction = nobs / N
seed = 3
rng = MersenneTwister(seed)
X, observed = draw_node_observations!(bp_obs, nsnaps, times = obs_times .+ 1, softinf=Inf; rng)


λinit = 0.1
ρinit = 0.1

A_complete = ones(N,N) - I
g_complete = IndexedGraph(A_complete)
λ_complete = sparse(λinit.*A_complete)
ρ_complete = fill(ρinit, N)

bp_inf = map(1:nsnaps-1) do i
    obs_sub = [sis.ϕ[a][(i-1)*separation+1:i*separation+1] for a in eachindex(sis.ϕ)]
    sis_inf = SIS_heterogeneous(g_complete, λ_complete, ρ_complete, separation; γ, ϕ=obs_sub)
    bp = mpbp(sis_inf)
end


svd_trunc = TruncBond(5)
maxiter = 30

nodes = vertices(bp_obs.g)

λder = [zeros(length(nodes)-1) for n in nodes]
ρder = zeros(length(nodes))
params_history = []

for it in 1:maxiter
    λstep = 0.5 * 3^(it≤(maxiter/4))
    ρstep = 0.5 * 3^(it≤(maxiter/4))
    for el in λder
        el .= 0.0
    end
    ρder .= 0.0

    cb = map(1:nsnaps-1) do k
        Threads.@threads for i in nodes
            onebpiter!(bp_inf[k], i, eltype(bp_inf[k].w[i]); svd_trunc, damp=0.0)
        end

        Threads.@threads for i in nodes
            λd, ρd, = MatrixProductBP.derivatives(bp_inf[k], i; svd_trunc, logpriorder=(x)->0.0)
            λder[i] .+= λd./(nsnaps-1)
            ρder[i] += ρd/(nsnaps-1)
        end
    end

    for k in 1:nsnaps-1
        for i in nodes
            wᵢ = bp_inf[k].w[i]
            dᵢ = length(inedges(bp_inf[k].g,i))
            for t in eachindex(wᵢ)
                wᵢ[t].λ .+= λstep .* erf.(0.5.*λder[i]) #.* wᵢ[t].λ
                wᵢ[t].ρ += ρstep * erf(0.5*ρder[i]) #* wᵢ[t].ρ
            end
            for t in eachindex(wᵢ)
                for j in 1:dᵢ
                    wᵢ[t].λ[j] = clamp(wᵢ[t].λ[j], 1e-6, 1-1e-6)
                end
                wᵢ[t].ρ = clamp(wᵢ[t].ρ, 1e-6, 1-1e-6)
            end
        end
    end

    data = MatrixProductBP.save_data(bp_inf[1])
    push!(params_history, data)
    
    jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/sigmoid_rand$(N)_snaps_step$(separation)_nobs$(nsnaps).jld2"; params_history, data, λ)
    println("iteration $(it) completed")
end