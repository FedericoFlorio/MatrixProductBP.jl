using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, PyPlot, DelimitedFiles
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using JLD2
include("/home/fedflorio/master_thesis/Utilities/correlations.jl")

A = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER.txt", Bool)
g = IndexedGraph(A)

@time for sep in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# for sep in [4]
    nsnaps = 100
    separation = sep
    T = nsnaps * separation
    N = nv(g)
    frac_edges = 0.6
    # n_edges_corr = Int(ceil(frac_edges*N)) - 1
    n_edges_corr = 6
    n_edges = 7*N
    # n_edges = ne(g)
    seed = 8    # !!! before running, test that the trajectory does not die out !!!

    λ_unif = 0.2
    ρ_unif = 0.2
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


    corr = correlations_traj(X[:,separation+1:separation:end])
    A_inf = zeros(N,N)
    for (i,c) in enumerate(corr)
        corr[i] = vcat(c[1:i-1],-Inf,c[i:end])
        p = sortperm(corr[i], rev=true)

        for n in 1:n_edges_corr

            A_inf[i,p[n]] = 1
            A_inf[p[n],i] = 1
        end
    end
    maxima_pos = argmax.(corr)
    maxima = [corr[i][m] for (i,m) in enumerate(maxima_pos)]
    for cnt in 1:n_edges
        m = argmax(maxima)
        maxima[m] == -Inf && break

        A_inf[m,maxima_pos[m]] = 1
        A_inf[maxima_pos[m],m] = 1

        corr[m][maxima_pos[m]] = -Inf
        maxima_pos[m] = argmax(corr[m])
        maxima[m] = corr[m][maxima_pos[m]]
    end
    g_inf = IndexedGraph(A_inf)
    neigs = [collect(neighbors(g_inf,i)) for i in vertices(g_inf)]

    # λinit = 1e-12
    # ρinit = 1e-12
    λinit = 0.01
    ρinit = 0.01

    # A_inf = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER_add.txt", Bool)
    # g_inf = IndexedGraph(A_inf)
    λ_inf = sparse(λinit.*A_inf)
    ρ_inf = fill(ρinit, N)

    sis_inf = SIS_heterogeneous(g_inf, λ_inf, ρ_inf, T; γ, ϕ=deepcopy(bp_obs.ϕ))
    bp_inf = mpbp(sis_inf)


    svd_trunc = TruncBond(5)
    maxiter = 30

    println("Inference on Erdos-Renyi with selected edges (according to correlations).\n\tλ = $(λ_unif)\t\tρ = $(ρ_unif)\n\tsnaps = $(nsnaps)\t\t step = $(separation)")
    params_history = []
    @time for it in 1:maxiter
        iters, cb = inference_parameters!(bp_inf, method=41, maxiter=1, λstep=0.01, ρstep=0.01)

        lam = [zeros(N) for _ in 1:N]
        for i in eachindex(lam)
            for (ii,j) in enumerate(neigs[i])
                lam[i][j] = cb.data[2].λ[i][ii]
            end
            lam[i] = vcat(lam[i][1:i-1],lam[i][i+1:end])
        end

        data = MatrixProductBP.PARAMS(lam,cb.data[2].ρ)

        # data = cb.data[2]
        push!(params_history,data)

        jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/simulations/ER30_select_lambda_$(Int(λ_unif*1000))_rho_$(Int(ρ_unif*1000))_step$(separation)_nobs$(nsnaps)_seed$(seed).jld2"; params_history, data, λ)
        println("iteration $(it) completed")
    end
end