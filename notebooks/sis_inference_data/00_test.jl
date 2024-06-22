using TensorTrains, LogarithmicNumbers, LinearAlgebra, MatrixProductBP, Tullio

function acc_L(A::MPEM3; normalize=true)
    Lt = [(m==n) for m in 1:size(A[begin],1), n in 1:size(A[begin],1), x in 1:size(A[begin],3)]
    z = Logarithmic(1.0)
    z_acc = zeros(Logarithmic, length(A))
    L = map(enumerate(A)) do (t,At)
        nt = maximum(abs, Lt)
        if !iszero(nt) && normalize
            Lt ./= nt
            z *= nt
        end
        t>1 && (z_acc[t-1] = z)
        @tullio Ltdummy[i,k,xᵗ⁺¹] := Lt[i,j,xᵗ] * At[j,k,xᵗ,xⱼ,xᵗ⁺¹]
        Lt = Ltdummy
    end
    z_acc[end] = Logarithmic(tr(Lt[:,:,1]))*z
    
    return L, z_acc
end

function acc_R(A::MPEM3; normalize=true)
    q = size(A[end],3)
    Rt = [(m==n)/q for m in 1:size(A[end],2), n in 1:size(A[end],2), x in 1:q]
    z = Logarithmic(1.0)
    T = length(A)
    z_acc = zeros(Logarithmic, T)
    R = map(enumerate(Iterators.reverse(A))) do (t,At)
        nt = maximum(abs, Rt)
        if !iszero(nt) && normalize
            Rt ./= nt
            z *= nt
        end
        t>1 && (z_acc[T+2-t] = z)
        @tullio Rtdummy[i,k,xᵗ] := At[i,j,xᵗ,xⱼ,xᵗ⁺¹] * Rt[j,k,xᵗ⁺¹]
        Rt = Rtdummy
    end |> reverse
    z_acc[1] = Logarithmic(sum(tr(Rt[:,:,x]) for x in axes(Rt,3))) * z

    return R, z_acc
end

tensors = [ rand(1,3,2,2,2), rand(3,4,2,2,2), rand(4,1,2,2,2) ]
tensors[end][:,:,:,:,2] .= tensors[end][:,:,:,:,1]
B = MPEM3(tensors)
A = mpem2(B)

(L,zL) = acc_L(B)
(R,zR) = acc_R(B)
L1, R2 = L[1], R[2]
@tullio M[i,k] := L1[i,j,x] * R2[j,k,x]
Z = only(M) * zL[1] * zR[2]

println(normalization(A))
println(zR[1])
println(zL[end])
println(Z)