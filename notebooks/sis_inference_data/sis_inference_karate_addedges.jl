using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, PyPlot, DelimitedFiles
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using JLD2

A = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/karate.txt", Bool)
g = IndexedGraph(A)

nsnaps = 400
separation = 16
T = nsnaps * separation
N = nv(g)
seed = 4

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
γ = [i==1 ? 1.0 : 0.0 for i in 1:N]

sis = SIS_heterogeneous(λ, ρ, T; γ)
bp_obs = mpbp(sis)

obs_times = collect(range(separation, step=separation, length=nsnaps))
nobs = floor(Int, N * length(obs_times) * 1.0)
obs_fraction = nobs / N
rng = MersenneTwister(seed)
X, observed = draw_node_observations!(bp_obs, nobs, times = obs_times .+ 1, softinf=Inf; rng)


λinit = 1e-12
ρinit = 1e-12

A_add = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/karate_add.txt", Bool)
g_add = IndexedGraph(A_add)
λ_add = sparse(λinit.*A_add)
ρ_add = fill(ρinit, N)

bp_inf = map(1:nsnaps-1) do i
    obs_sub = [sis.ϕ[a][(i-1)*separation+1:i*separation+1] for a in eachindex(sis.ϕ)]
    sis_inf = SIS_heterogeneous(g_add, λ_add, ρ_add, separation; γ, ϕ=obs_sub)
    bp = mpbp(sis_inf)
end


svd_trunc = TruncBond(5)
maxiter = 30

nodes = vertices(bp_obs.g)

λder = [zeros(length(neighbors(g_add,n))) for n in nodes]
ρder = zeros(length(nodes))
params_history = []

for it in 1:maxiter
    λstep = 0.01 * 3^(it≤(maxiter/4))
    ρstep = 0.01 * 3^(it≤(maxiter/4))
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
                wᵢ[t].λ .+= λstep .* tanh.(0.5.*λder[i]) #.* wᵢ[t].λ
                wᵢ[t].ρ += ρstep * tanh(0.5*ρder[i]) #* wᵢ[t].ρ
            end
            for t in eachindex(wᵢ)
                for j in 1:dᵢ
                    wᵢ[t].λ[j] = clamp(wᵢ[t].λ[j], 1e-9, 1-1e-9)
                end
                wᵢ[t].ρ = clamp(wᵢ[t].ρ, 1e-9, 1-1e-9)
            end
        end
    end

    data = MatrixProductBP.save_data(bp_inf[1])
    push!(params_history, data)
    
    jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/sigmoid_karate_add_snaps_step$(separation)_nobs$(nsnaps).jld2"; params_history, data, λ)
    println("iteration $(it) completed")
end