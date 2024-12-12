using TensorTrains, Tullio, TensorCast, OffsetArrays, LogarithmicNumbers
using UnPack: @unpack
using InvertedIndices: Not
using HypergeometricFunctions: _₂F₁
import MatrixProductBP: FourierMPEM2
# using JLD2

potts2spin(x; q=2) = (x-1)/(q-1)*2 - 1
spin2potts(σ; q=2) = (σ+1)/2*(q-1) + 1
# potts2spin(x) = 3-2x
# spin2potts(σ) = (3-σ)/2

"""
    Fourier3int(α::Int, β::Int, γ::Int; w₁::Float64=0.5, w₂::Float64=0.5, J1::Float64=1.0, J2::Float64=1.0, P::Float64=2.0)

Computes the integral ``∫dx₁ ∫dx₂ F_α(x₁) F_β(x₂) F*_γ(J₁x₁+J₂x₂)``. The integrations ranges are ``[-w₁,+w₁]`` for the integral in ``dx₁`` and ``[-w₂,+w₂]`` for the integral in ``dx₂``
"""
function Fourier3int(α::Int, β::Int, γ::Int, w₁::Real, w₂::Real, J1::Float64, J2::Float64, P::Float64)
    kαγ = 2π/P*(α-J1*γ)
    kβγ = 2π/P*(β-J2*γ)
    I1 = iszero(kαγ) ? w₁ : sin(kαγ*w₁)/kαγ
    I2 = iszero(kβγ) ? w₂ : sin(kβγ*w₂)/kβγ

    return 4/P * I1 * I2
end

function op(F₁::FourierMPEM2{U1}, F₂::FourierMPEM2{U1}, J₁::U2, J₂::U2, K::Int, w₁::Real, w₂::Real, P::Float64, svd_trunc; normalize::Bool=true) where {U1<:Number, U2<:Real}
    K1 = (size(F₁[1],3)-1)/2 |> Int
    K2 = (size(F₂[1],3)-1)/2 |> Int
    Int_tensor_ = [Fourier3int(α,β,γ, w₁,w₂, J₁,J₂, P) for α=-K1:K1, β=-K2:K2, γ=-K:K]
    Int_tensor = OffsetArray(Int_tensor_, -K1:K1, -K2:K2, -K:K)
    any(isnan, Int_tensor) && @error "NaN in Fourier integral"

    GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
        any(isnan, F₁ᵗ) && error("NaN in F₁ᵗ")
        any(isnan, F₂ᵗ) && error("NaN in F₂ᵗ")
        @tullio Gt[m1,m2,n1,n2,γ,x] := F₁ᵗ[m1,n1,α,x] * F₂ᵗ[m2,n2,β,x] * Int_tensor[α,β,γ]
        @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
        any(isnan, Gᵗ) && error("NaN in convolution")
        return collect(Gᵗ)
    end

    G = FourierTensorTrain(GG, z=F₁.z*F₂.z)
    compress!(G; svd_trunc)
    normalize && normalize_eachmatrix!(G)
    return G
end

function convolution(F::Vector{FourierMPEM2{U1}}, J::Vector{U2}, P::Float64;
    K::Int=(size(F[1][1],3)-1)/2, scale::Real=1.0, svd_trunc=TruncThresh(1e-8)) where {U1<:Number, U2<:Real}
    D = length(F)
    D==1 && return F[1]
    w = 1/scale # = P/2D

    G = op(F[1], F[2], J[1], J[2], K, w, w, P, svd_trunc)
    for i in 3:D
        G = op(G, F[i], 1.0, J[i], K, (i-1)*w, w, P, svd_trunc)
    end
    return G
end



function _compute_integral(β,Jⱼᵢ,xⱼᵗ,hᵢ,xᵢᵗ⁺¹,kᵧ, y, s)
    Jxj = β * (Jⱼᵢ*xⱼᵗ + hᵢ)
    if iszero(kᵧ) && xᵢᵗ⁺¹==-1
        Int_ind = (2β .* (Jxj.+s.*y) .- log.(β*s .* (1 .+ exp.(2β .* (Jxj.+s.*y))))) ./ (2β*s)
    else
        xp1 = 1+xᵢᵗ⁺¹
        bb = (xp1 + 1im*kᵧ/(β*s)) / 2
        cc = 1 + bb
        denom = (kᵧ-1im*β*s*xp1)
        expon = exp.(1im*kᵧ .* y .+ β.*xp1.*(Jxj.+s.*y))
        hypgeom = map(X -> _₂F₁(1.0, bb, cc, -exp(2β*(Jxj+s*X))), y)
        Int_ind = -1im .* expon ./ denom .* hypgeom
    end

    return Int_ind[2] - Int_ind[1]
end

function onebpiter_fourier!(bp::MPBP, i::Integer, K::Integer; P=2.0, σ=1/50, svd_trunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i][1], ϕ[i], length(ein)
    J, hᵢ, β = float.(wᵢ.J), wᵢ.h, wᵢ.β
    scale = dᵢ+2

    μ_fourier = [FourierTensorTrain_spin(μ[k], K, scale, P, σ) for k in idx.(ein)]
    for (j_ind,e_out) in enumerate(eout)
        notj = eachindex(μ_fourier)[Not(j_ind)]
        if isempty(notj)
            μj = map(eachindex(bp.μ[idx(e_out)])) do t
                Aᵗ = zeros(1,1,2,2,2)
                @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = exp(β*J[j_ind]*potts2spin(xⱼᵗ)*potts2spin(xᵢᵗ⁺¹)) / 2cosh(β*J[j_ind]*potts2spin(xⱼᵗ))
            end |> MPEM3 |> mpem2

            compress!(μj; svd_trunc)
            normalize!(μj)
            bp.μ[idx(e_out)] = μj
        else
            conv_μ_notj = convolution(μ_fourier[notj], J[notj], P; K, scale, svd_trunc)
            @tullio In[γ,xᵢᵗ⁺¹,xⱼᵗ] := _compute_integral(β,J[j_ind],potts2spin(xⱼᵗ),hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1,1], scale) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2
            any(isnan, In) && error("NaN in integral")

            μj = map(eachindex(conv_μ_notj)) do t
                μᵗ₋ⱼ, ϕᵢᵗ = conv_μ_notj[t], ϕᵢ[t]
                @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μᵗ₋ⱼ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹,xⱼᵗ] * ϕᵢᵗ[xᵢᵗ]
                real.(Aᵗ)
            end
            μᵀ₋ⱼ, ϕᵢᵀ = conv_μ_notj[end], ϕᵢ[end]
            @tullio μjᵀ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μᵀ₋ⱼ[m,n,0,xᵢᵗ] * P * ϕᵢᵀ[xᵢᵗ] xⱼᵗ∈1:2, xᵢᵗ⁺¹∈1:2
            μj[end] = real.(μjᵀ)
            μᵢⱼ = collect.(μj) |> MPEM3 |> mpem2

            compress!(μᵢⱼ; svd_trunc)
            normalize!(μᵢⱼ)
            bp.μ[idx(e_out)] = μᵢⱼ
        end
    end

    conv_μ_full = convolution(μ_fourier, J, P; K, scale, svd_trunc)
    @tullio In[γ,xᵢᵗ⁺¹] := _compute_integral(β,0.0,0.0,hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1,1], scale) γ∈-K:K, xᵢᵗ⁺¹∈1:2
    any(isnan, In) && error("NaN in integral")
    b = map(eachindex(conv_μ_full)) do t
        μ_fullᵗ, ϕᵢᵗ = conv_μ_full[t], ϕᵢ[t]
        @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μ_fullᵗ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹] * ϕᵢᵗ[xᵢᵗ] xⱼᵗ∈1:2
        real.(Aᵗ)
    end
    μ_fullᵀ, ϕᵢᵀ = conv_μ_full[end], ϕᵢ[end]
    @tullio bT[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μ_fullᵀ[m,n,0,xᵢᵗ] * P * ϕᵢᵀ[xᵢᵗ] xⱼᵗ∈1:2, xᵢᵗ⁺¹∈1:2
    b[end] = real.(bT)
    belief = collect.(b) |> MPEM3 |> mpem2 |> marginalize

    normalize!(belief)
    bp.b[i] = belief

    return nothing
end

function iterate_fourier!(bp::MPBP, K::Integer; maxiter::Integer=5, svd_trunc::SVDTrunc=TruncThresh(1e-6), nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, σ::Real=1/50)
    for it in 1:maxiter
        # Threads.@threads for i in nodes
        for i in nodes
            onebpiter_fourier!(bp, i, K; svd_trunc)
        end
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return nothing
end