using Pkg
Pkg.activate("/home/fedflorio/master_thesis/")

using DelimitedFiles, JLD2, Graphs, IndexedGraphs

N = 100
c = 2.5
gg = erdos_renyi(N, c/N)
gg = IndexedGraph(gg)
cnt = 0
nedges = ne(gg)
A = gg.A
A = Int.(map(!=(0), A))

println("Original matrix")
display(A)

open("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER100.txt", "w") do io
    writedlm(io, A)
end

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

open("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER100_add.txt", "w") do io
    writedlm(io, A)
end

jldsave("/home/fedflorio/master_thesis/MatrixProductBP.jl/notebooks/sis_inference_data/graphs/ER100_add_neigs.jld2"; neigs)

println("Modified matrix")
display(A)