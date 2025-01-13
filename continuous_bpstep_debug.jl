using Pkg
Pkg.activate("C:/Users/fefif/Desktop/PhD/Progetti/MatrixProductBP.jl")

using Revise
using MatrixProductBP, MatrixProductBP.Models
using Graphs, IndexedGraphs, Statistics, Random, LinearAlgebra, TensorTrains, SparseArrays
import ProgressMeter; ProgressMeter.ijulia_behavior(:clear)
using TensorTrains: summary_compact
using SparseArrays
using JLD2
include("bp_fourier.jl");

seed = 1
rng = MersenneTwister(seed)

T = 4
N = 3
A = [0 1 1; 
     1 0 0;
     1 0 0;]
# N = 4
# A = [0 1 0 0; 
#      1 0 1 0; 
#      0 1 0 1; 
#      0 0 1 0]
# N = 7
# A = [0 1 0 0 0 0 0; 
#      1 0 1 0 0 0 0; 
#      0 1 0 1 0 0 0;
#      0 0 1 0 1 0 0;
#      0 0 0 1 0 1 0;
#      0 0 0 0 1 0 1;
#      0 0 0 0 0 1 0]
# N = 4
# A = [0 1 1 1;
#      1 0 0 1;
#      1 0 0 1;
#      1 1 1 0]
# !!! Here the calculations are wrong for edge 1->2 !!!
# N = 5
# A = [0 1 1 1 1; 
#      1 0 0 0 0; 
#      1 0 0 0 0; 
#      1 0 0 0 0; 
#      1 0 0 0 0]
# N = 6
# A = [0 1 1 1 0 0; 
#      1 0 0 0 1 0; 
#      1 0 0 0 0 1; 
#      1 0 0 0 0 0; 
#      0 1 0 0 0 0; 
#      0 0 1 0 0 0]
# N = 7
# A = [0 1 1 1 1 0 1;
#      1 0 0 0 1 0 0; 
#      1 0 0 0 0 1 0; 
#      1 0 0 0 0 0 0; 
#      1 1 0 0 0 0 0; 
#      0 0 1 0 0 0 0; 
#      1 0 0 0 0 0 0]
# N = 8
# A = [0 1 1 1 0 0 1 0;
#      1 0 0 0 1 0 0 1; 
#      1 0 0 0 0 1 0 0; 
#      1 0 0 0 0 0 0 0; 
#      0 1 0 0 0 0 0 0; 
#      0 0 1 0 0 0 0 0; 
#      1 0 0 0 0 0 0 0;
#      0 1 0 0 0 0 0 0]

g = IndexedBiDiGraph(A)

# N = 20
# c = 3
# gg = erdos_renyi(N, c/N; seed)
# g = IndexedBiDiGraph(gg)
# A = zeros(Int,N,N)
# for i in CartesianIndices(A)
#      if !iszero(g.A[i])
#          A[i] = 1
#      end
# end

β = 1.0
J = 1
h = 0.0
m⁰ = 1.0
K = 100

svd_trunc=TruncBond(4)

A

w = [fill(IntegerGlauberFactor([J*A[j,i] for j in inneighbors(g,i)], h, β), T+1) for i in vertices(g)]
ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
bp = mpbp(g, w, fill(2,N), T; ϕ=fill(ϕᵢ,N))

for μ in bp.μ
    for a in μ
        a.=rand.(rng)
        # a.=1.0
    end
end
# bp_fourier = deepcopy(bp)
iterate!(bp, maxiter=1; svd_trunc)

msg = deepcopy(bp.μ)
# jldsave("messages.jld2"; msg)

bp_fourier = deepcopy(bp)
# iterate_fourier!(bp_fourier,K, maxiter=1, σ=1/100; svd_trunc)
;

iterate!(bp,maxiter=1; svd_trunc)

iterate_fourier!(bp_fourier,K, maxiter=1, σ=1/50; svd_trunc)
