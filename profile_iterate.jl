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
N = 4
A = [0 1 1 1;
     1 0 0 1;
     1 0 0 1;
     1 1 1 0]
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

g = IndexedBiDiGraph(A)
# N = 10
# c = 3
# gg = erdos_renyi(N, c/N; seed)
# g = IndexedBiDiGraph(gg)
# A = zeros(Int,N,N)
# for i in CartesianIndices(A)
#      if !iszero(g.A[i])
#          A[i] = 1
#      end
# end
k = 4
# rand!(rng, A.nzval, (-1,1))  # fill directed incidence matrix with rand(-1,1)
β = 1.0
J = 1
h = 0.0
m⁰ = 1.0
K = 50

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
iterate!(bp, maxiter=1; svd_trunc)
# msg = deepcopy(bp.μ)
# jldsave("messages.jld2"; msg)

bp_fourier = deepcopy(bp)
;
iterate!(bp,maxiter=1; svd_trunc)

using Profile
@profview iterate_fourier!(bp_fourier,K, maxiter=1, σ=1/50; svd_trunc)