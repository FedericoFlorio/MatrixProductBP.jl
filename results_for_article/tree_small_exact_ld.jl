using Pkg
Pkg.activate(".")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays, Tullio, ProgressMeter, OffsetArrays, JLD2
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact

seed = 1
rng = MersenneTwister(seed)

T = 10
N = 10
gg = barabasi_albert(N, 2, 1; rng, complete=true)
g = IndexedBiDiGraph(gg)

β = 0.5
h = 0.0
m⁰ = 0.6
p = 0.0
P = 2.0

J = zeros(nv(g),nv(g))
for i in axes(J)[1], j in axes(J)[2]
    # j>i && continue
    if has_edge(gg,i,j)
        J[i,j] = rand(rng)
        # J[i,j] = rand(rng)
        # J[i,j] = 0.0
        J[j,i] = J[i,j]
    end
end

d = 15
K = 30
σ = 1/30

ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
ψ_neutral = [ones(2,2) for t in 0:T]
ϕ = fill(ϕᵢ, nv(g))
w_fourier = [fill(FourierGlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β; K, σ, P, p), T+1) for i in vertices(g)]

maxiter = 20
tol = 1e-10

hT = 1e-4 * (-5N)
ϕᵢ[end] = [exp(-hT), exp(hT)]

bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)

bp_h = map(1e-4 .* (-5N:5N), -5N:5N) do hT, cnt
    println(hT)

    ϕᵢ[end] = [exp(-hT), exp(hT)]
    w_fourier = [fill(FourierGlauberFactor([J[ed.src,ed.dst] for ed in inedges(g,i)], h, β; K, σ, P, p), T+1) for i in vertices(g)]
    
    maxiter = 20
    tol = 1e-10
    bp_fourier = mpbp(ComplexF64, g, w_fourier, fill(2, nv(g)), T; ϕ)
    
    iters_fourier, cb_fourier = iterate!(bp_fourier; maxiter, svd_trunc=TruncBond(d), tol)
    m_fourier = real.(means(potts2spin, bp_fourier))
    
    jldsave("tree_small_bp_ld_$(N)_beta0,5_h0_randomJ_$(cnt).jld2"; bp_fourier.f, m_fourier)
    bp_fourier
end



function state_ising(x::Int64, N::Int64)
    map(1:N) do i
        (((x >> (i - 1)) & 1) == 0) ? -1 : 1
    end
end

spin(x, i) = (((x >> (i - 1)) & 1) == 0) ? -1 : 1

p0_ising(state, N) = prod((1+m⁰)/2*(spin(state,i)==1) + (1-m⁰)/2*(spin(state,i)==-1) for i in 1:N)

f(x,H) = 1/(1+exp(-2*H*x))

function ptrans_ising(statenew, stateold, N, J, h, β, p)
    prod((H = β*(h + sum(J[ed.src,i]*spin(stateold, ed.src) for ed in inedges(g,i))); p*(spin(statenew, i)== spin(stateold,i)) + (1-p)*f(spin(statenew, i), H)) for i in 1:N)
end

function compute_exact_probabilities(prob_ising, trans_matrix, T)
    @showprogress map(1:T) do _
        prob_ising = trans_matrix * prob_ising
    end
end

function compute_magnetizations(prob_exact, N, T)
    m = [Vector{Float64}(undef, T+1) for _ in 1:N]
    @showprogress for t in 1:(T+1)
        for i in 1:N
            m[i][t] = sum(prob_exact[t][x] * spin(x,i) for x in 1:2^N)
        end
    end
    return m
end

function large_deviations(prob_exact, N, t)
    p = prob_exact[t]
    pm = OffsetVector(zeros(2N+1), -N:N)
    @showprogress for x in 1:2^N
        m = sum(spin(x,i) for i in 1:N)
        pm[m] += p[x]
    end
    pm
end

prob_ising = @showprogress [p0_ising(x,N) for x in 1:2^N]
trans_matrix = @showprogress [ptrans_ising(y, x, N, J, h, β, p) for y in 1:2^N, x in 1:2^N]
prob_exact = vcat([prob_ising], compute_exact_probabilities(prob_ising, trans_matrix, T))
large_dev_exact = [large_deviations(prob_exact, N, t) for t in 1:(T+1)]

jldsave("tree_small_exact_ld_$(N)_beta0,5_h0_randomJ.jld2"; large_dev_exact)