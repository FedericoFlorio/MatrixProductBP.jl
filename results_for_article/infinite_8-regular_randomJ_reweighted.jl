using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Distributions
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
import ProgressMeter: @showprogress
using TensorTrains: summary_compact
using JLD2

T = 10
k = 8   # change J
m⁰ = 0.6
β = 1.0
K = 30
σ = 1/30
P = 2.0
p = 0.0
prob_degree = Dirac(k)

popsize = 10^2

function prob_w(w; d=dᵢ, rng=Xoshiro(1))
    T = length(w) - 1

    prob_J = Uniform(0.0, 1/7)
    prob_h = Dirac(0.0)
    Js = rand(rng, prob_J, d)
    h = rand(prob_h)

    fill(FourierGlauberFactor(Js,h,β; K,σ,P,p), T+1)
end

w_init = fill(FourierGlauberFactor([0.0],0.0,1.0; K,σ,P,p), T+1)

ϕᵢ = [ones(2) for t in 0:T]
ϕᵢ[1] = [(1-m⁰)/2, (1+m⁰)/2]
ψ_neutral = [ones(2,2) for t in 0:T]

function stats!((bs, fs, ers), wᵢ, μ, μin, b, f)
    push!(bs, [real.(m) for m in marginals(b)])
    push!(fs, f)

    mo = marginals.(μ)
    mn = marginals.(μin)
    er = maximum(maximum(abs.(m1.-m2)) for (m1,m2) in zip(mn[1],mo[1]))
    push!(ers, er)
end

μ_pop = map(1:popsize) do p
    μ = rand_mpem2(ComplexF64, 2, 2, T)
    normalize!(μ)
    μ
end |> AtomicVector

# Initialization
h = -0.24
ϕᵢ[end] = [exp(-h), exp(h)]
bonddims = [5, 10]
maxiters = [2000, 4000]

for ind in eachindex(bonddims)
    d = bonddims[ind]
    maxiter = maxiters[ind]

    bs = Vector{Vector{Float64}}[] |> AtomicVector
    iterate_popdyn!(μ_pop, w_init, prob_degree, prob_w, (bs,); ϕ=ϕᵢ, T, maxiter, svd_trunc=TruncBond(d))
end


# Iteration
d = 10
maxiter = 2000
cnt = -27
msg_pops = map(-0.24:0.03:0.24) do h
    global cnt += 3
    println(h)
    ϕᵢ[end] = [exp(-h), exp(h)]

    bs = Vector{Vector{Float64}}[] |> AtomicVector
    fs = Float64[] |> AtomicVector
    ers = Float64[] |> AtomicVector

    iterate_popdyn!(μ_pop, w_init, prob_degree, prob_w, (bs, fs, ers); ϕ=ϕᵢ, stats=stats!, T, maxiter, svd_trunc=TruncBond(d))

    jldsave("article/popdyn_infinite_8-reg_reweight_h$(cnt)_dmax$(d)_K$(K).jld2"; bs, fs, ers)
    μ_pop
end
;