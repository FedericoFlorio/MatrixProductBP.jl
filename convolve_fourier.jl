using TensorTrains, Tullio, TensorCast, OffsetArrays, LogarithmicNumbers
using UnPack: @unpack
using InvertedIndices: Not
using HypergeometricFunctions: _₂F₁
# using JLD2

potts2spin(x; q=2) = (x-1)/(q-1)*2 - 1
spin2potts(σ; q=2) = (σ+1)/2*(q-1) + 1
# potts2spin(x) = 3-2x
# spin2potts(σ) = (3-σ)/2

"""
    Fourier3int(α::Int, β::Int, γ::Int; w₁::Float64=0.5, w₂::Float64=0.5, J1::Float64=1.0, J2::Float64=1.0, P::Float64=2.0)

Computes the integral ``∫dx₁ ∫dx₂ F_α(x₁) F_β(x₂) F*_γ(J₁x₁+J₂x₂)``. The integrations ranges are ``[-w₁,+w₁]`` for the integral in ``dx₁`` and ``[-w₂,+w₂]`` for the integral in ``dx₂``
"""
function Fourier3int(α::Int, β::Int, γ::Int, w₁::Float64, w₂::Float64, J1::Float64, J2::Float64, P::Float64)
    kαγ = 2π/P*(α-J1*γ)
    kβγ = 2π/P*(β-J2*γ)
    I1 = iszero(kαγ) ? w₁ : sin(kαγ*w₁)/kαγ
    I2 = iszero(kβγ) ? w₂ : sin(kβγ*w₂)/kβγ

    return 4/P * I1 * I2
end

function op(F₁::FourierTensorTrain{U1,N}, F₂::FourierTensorTrain{U1,N}, J₁::U2, J₂::U2, K::Int, w₁::Float64, w₂::Float64, P::Float64, ax::Int64, svd_trunc; normalize::Bool=true) where {U1<:Number, U2<:Real, N}
    K1 = (size(F₁[1])[ax]-1)/2 |> Int
    K2 = (size(F₂[1])[ax]-1)/2 |> Int
    Int_tensor = OffsetArray([Fourier3int(α,β,γ, w₁,w₂, J₁,J₂, P) for α=-K1:K1, β=-K2:K2, γ=-K:K], -K1:K1, -K2:K2, -K:K)
    any(isnan, Int_tensor) && @error "NaN in Fourier integral"
    GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
        any(isnan, F₁ᵗ) && error("NaN in F₁ᵗ")
        any(isnan, F₂ᵗ) && error("NaN in F₂ᵗ")
        F1t_ = reshape(F₁ᵗ, size(F₁ᵗ)[1:ax]..., prod(size(F₁ᵗ)[ax+1:end]))
        F1t = OffsetArray(F1t_, axes(F1t_)[1:ax-1]..., -K1:K1, axes(F1t_)[ax+1])
        F2t_ = reshape(F₂ᵗ, size(F₂ᵗ)[1:ax]..., prod(size(F₂ᵗ)[ax+1:end]))
        F2t = OffsetArray(F2t_, axes(F2t_)[1:ax-1]..., -K2:K2, axes(F2t_)[ax+1])
        @tullio Gt[m1,m2,n1,n2,γ,x] := F1t[m1,n1,α,x] * F2t[m2,n2,β,x] * Int_tensor[α,β,γ]
        @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
        any(isnan, Gᵗ) && error("NaN in convolution")
        return collect(Gᵗ)
    end

    G = FourierTensorTrain(GG, z=F₁.z*F₂.z)
    compress!(G; svd_trunc)
    normalize && normalize_eachmatrix!(G)
    return G
end

function convolution(F::Vector{FourierTensorTrain{U1,N}}, J::Vector{U2}, P::Float64, ax::Int64;
    K::Int=(size(F[1][1])[ax]-1)/2, svd_trunc=TruncThresh(1e-8)) where {U1<:Number, U2<:Real, N}
    D = length(F)
    D==1 && return F[1]
    w = P/2D

    G = op(F[1], F[2], J[1], J[2], K, w, w, P, ax, svd_trunc)
    for i in 3:D
        G = op(G, F[i], 1.0, J[i], K, (i-1)*w, w, P, ax, svd_trunc)
    end
    return G
end



function compute_integral(yextrema,β,Jⱼᵢ,hᵢ,xᵢᵗ⁺¹,xⱼᵗ,γ,P)
    if γ==0 && xᵢᵗ⁺¹==-1
        ind_int = -1/(2β) .* log.(1 .+ exp.(-2β .* (yextrema .+ Jⱼᵢ*xⱼᵗ .+ hᵢ)))
        any(isnan, ind_int) && error("NaN in ind_int")
    else
        ikγ = 1im * 2π/P * γ / yextrema[2]  # the / yextrema[2] accounts for the rescaling of the field done during the convolution
        Jxj = Jⱼᵢ*xⱼᵗ + hᵢ
        x = 1+xᵢᵗ⁺¹
        wave = exp.(ikγ.*yextrema)
        denom = β*x + ikγ
        iszero(denom) && error("Zero in denominator")
        expon = exp.(β.*x .* (J .+ yextrema))
        a = 1
        b = x/2 + ikγ/(2*β)
        c = 1 + b
        z = -exp.(2*β .* Jxj .+ yextrema)
        hypgeom = map(X -> _₂F₁(a,b,c,X), z)
        any(isnan, hypgeom) && error("NaN in Hypergeometric function")
        ind_int = wave ./ denom .* expon .* hypgeom
        any(isnan, ind_int) && error("NaN in ind_int")
    end
    
    return ind_int[2] - ind_int[1]
end

function onebpiter_fourier!(bp::MPBP, i::Integer, K::Integer; P=2.0, σ=1/50)
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
                trans_prob(xᵢᵗ⁺¹,xⱼᵗ) = exp(β*J[j_ind]*potts2spin(xⱼᵗ)*potts2spin(xᵢᵗ⁺¹)) / 2cosh(β*J[j_ind]*xⱼᵗ)
                Aᵗ = zeros(1,1,2,2,2)
                @tullio Aᵗ[m,n,xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ] = trans_prob(xᵢᵗ⁺¹,xⱼᵗ)
            end |> MPEM3 |> mpem2
            bp.μ[idx(e_out)] = μj
        else
            conv_μ_notj = convolution(μ_fourier[notj], J[notj], P, 3, K=K)

            @tullio In_[γ,xᵢᵗ⁺¹,xⱼᵗ] := compute_integral([-scale+1,scale-1], β, J[j_ind], hᵢ, potts2spin(xᵢᵗ⁺¹), potts2spin(xⱼᵗ), γ, P) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2
            In = OffsetArray(In_, -K:K, 1:2, 1:2)
            any(isnan, In) && error("NaN in integral")

            μj = map(eachindex(conv_μ_notj)) do t
                μᵗ₋ⱼ, ϕᵢᵗ = conv_μ_notj[t], ϕᵢ[t]
                @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μᵗ₋ⱼ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹,xⱼᵗ] * ϕᵢᵗ[xᵢᵗ]
                collect(real.(Aᵗ))
            end |> MPEM3 |> mpem2

            bp.μ[idx(e_out)] = μj
        end
    end

    conv_μ_full = convolution(μ_fourier, J, P, 3, K=K)
    @tullio In_[γ,xᵢᵗ⁺¹,xⱼᵗ] := compute_integral([-scale,scale], β, 0.0, hᵢ, potts2spin(xᵢᵗ⁺¹), potts2spin(xⱼᵗ), γ, P) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2
    In = OffsetArray(In_, -K:K, 1:2, 1:2)
    any(isnan, In) && error("NaN in integral")
    b = map(eachindex(conv_μ_full)) do t
        μ_fullᵗ, ϕᵢᵗ = conv_μ_full[t], ϕᵢ[t]
        @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μ_fullᵗ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹,xⱼᵗ] * ϕᵢᵗ[xᵢᵗ]
        collect(real.(Aᵗ))
    end |> MPEM3 |> mpem2 |> marginalize

    bp.b[i] = b

    return nothing
end

function iterate_fourier!(bp::MPBP, K::Integer; maxiter::Integer=5, svd_trunc::SVDTrunc=TruncThresh(1e-6), nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true)
    for it in 1:maxiter
        # Threads.@threads for i in nodes
        for i in nodes
            onebpiter_fourier!(bp, i, K)
        end
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return nothing
end