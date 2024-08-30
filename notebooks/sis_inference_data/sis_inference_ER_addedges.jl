using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, PyPlot, DelimitedFiles
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using JLD2

A = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER.txt", Bool)
g = IndexedGraph(A)

for sep in [1, 2, 4, 8, 16]
    seeds = [1]
    for seed in seeds
    ###############################################
        nsnaps = 100
        separation = sep
        T = nsnaps * separation
        N = nv(g)
        # seed = 1    # !!! before running, test that the trajectory does not die out !!!

        λ_unif = 0.2
        ρ_unif = 0.175
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
        γ = 0.1

        sis = SIS_heterogeneous(λ, ρ, T; γ)
        bp_obs = mpbp(sis)

        obs_times = collect(range(separation, step=separation, length=nsnaps))
        nobs = floor(Int, N * length(obs_times) * 1.0)
        obs_fraction = nobs / N
        rng = MersenneTwister(seed)
        X, observed = draw_node_observations!(bp_obs, nobs, times = obs_times .+ 1, softinf=Inf; rng)


        # λinit = 1e-12
        # ρinit = 1e-12
        λinit = 0.01
        ρinit = 0.01

        A_add = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER_add.txt", Bool)
        g_add = IndexedGraph(A_add)
        λ_add = sparse(λinit.*A_add)
        ρ_add = fill(ρinit, N)

        sis_inf = SIS_heterogeneous(g_add, λ_add, ρ_add, T; γ, ϕ=deepcopy(bp_obs.ϕ))
        bp_inf = mpbp(sis_inf)


        svd_trunc = TruncBond(5)
        maxiter = 30

        println("Inference on Erdos-Renyi with added edges.\n\tλ = $(λ_unif)\t\tρ = $(ρ_unif)\n\tsnaps = $(nsnaps)\t\t step = $(separation)")
        params_history = []
        @time for it in 1:maxiter
            iters, cb = inference_parameters!(bp_inf, method=41, maxiter=1, λstep=0.01, ρstep=0.01)

            data = cb.data[2]
            push!(params_history,data)

            jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/simulations/ER_add_lambda_$(Int(λ_unif*1000))_rho_$(Int(ρ_unif*1000))_step$(separation)_nobs$(nsnaps)_seed$(seed).jld2"; params_history, data, λ)
            println("iteration $(it) completed")
        end

    ##############################################
    end
end