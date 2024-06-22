using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using DelimitedFiles, JLD2, IndexedGraphs

A = readdlm("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/karate.txt", Int)
cnt = 0
N = 34
nedges = 78
while true
    global cnt, nedges
    i = rand(1:N)
    j = rand(1:N)
    if i≠j && A[i,j]==0
        A[i,j] = 1
        A[j,i] = 1
        cnt += 1
    end
    cnt == nedges && break
end

g = IndexedGraph(A)
neigs = [collect(neighbors(g,i)) for i in vertices(g)]

open("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/karate_add.txt", "w") do io
    writedlm(io, A)
end

jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/karate_add_neigs.jld2"; neigs)

A