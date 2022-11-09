import Random

# maps (1,2) -> (1,-1)
potts2spin(x) = 3-2x
spin2potts(σ) = (3+σ)/2

struct AllOneTensor; end
Base.getindex(::AllOneTensor, idx...) = 1
Base.size(::AllOneTensor, i::Integer) = 1
Base.axes(::AllOneTensor, i::Integer) = 1:1

# compute the kronecker product only over needed indices
kron2() = AllOneTensor()
function kron2(A₁::Array{F,4}) where F
    @cast _[m₁, n₁, xᵢ, x₁] := A₁[m₁, n₁, x₁, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}) where F
    # this is in case A₂ has a wider range for xᵢ.
    # that only happens when A₂ has the same value no matter xᵢ, so we might
    #  as well truncate it
    q = size(A₁)[4]
    A₂ = A₂[:,:,:,1:q]
    @cast _[(m₁, m₂), (n₁, n₂), xᵢ, x₁, x₂] := A₁[m₁, n₁, x₁, xᵢ] * 
        A₂[m₂, n₂, x₂, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃), (n₁, n₂, n₃), xᵢ, x₁, x₂, x₃] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ]
end

# symmetrize and set diagonal to zero
function symmetrize_nodiag!(A::AbstractMatrix)
    A .= (A+A')/2
    for i in axes(A, 1)
        A[i,i] = 0
    end
    A
end

# SAMPLING
# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoid creating a `Weight` object
function sample_noalloc(rng::Random.AbstractRNG, w::AbstractVector) 
    t = rand(rng) * sum(w)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
sample_noalloc(w::AbstractVector) = sample_noalloc(Random.GLOBAL_RNG, w)

# first turn integer `x` into its binary representation, then reshape the
#  resulting bit vector into a matrix of size specified by `dims`
function int_to_matrix(x::Integer, dims)
    y = digits(x, base=2, pad=prod(dims))
    return reshape(y, dims) .+ 1
end