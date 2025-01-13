using TensorTrains, Tullio, TensorCast, OffsetArrays, LogarithmicNumbers, CavityTools
using Unzip: unzip
using UnPack: @unpack
using ProgressMeter: Progress, ProgressUnknown, next!
using InvertedIndices: Not
using HypergeometricFunctions: _₂F₁
import MatrixProductBP: FourierMPEM2, FourierMPEM1

# using Memoization
using JLD2
using Infiltrator

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

function Fourier3int_1(α::Int, γ::Int, w::Real, J::Float64, P::Float64)
    kαγ = 2π/P*(α-J*γ)
    iszero(kαγ) ? 2w : sin(kαγ*w)/kαγ
end

function convolution(F::Vector{FourierMPEM2{U1}}, J::Vector{U2}, P::Float64;
    K::Int=(size(F[1][1],3)-1)/2, scale::Real=1.0, svd_trunc=TruncThresh(1e-8), normalize::Bool=true) where {U1<:Number, U2<:Real}

    function op((F₁, J₁, d₁), (F₂, J₂, d₂))
        K1 = (size(F₁[1],3)-1)/2 |> Int
        K2 = (size(F₂[1],3)-1)/2 |> Int
        @tullio Int_1[γ,α] := Fourier3int_1(α,γ,max(1,d₁)/scale,J₁,P) α∈-K1:K1, γ∈-K:K
        @tullio Int_2[γ,β] := Fourier3int_1(β,γ,max(1,d₂)/scale,J₂,P) β∈-K2:K2, γ∈-K:K
    
        GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
            @tullio Gt1[m1,n1,γ,x] := F₁ᵗ[m1,n1,α,x] * Int_1[γ,α]
            @tullio Gt2[m2,n2,γ,x] := F₂ᵗ[m2,n2,β,x] * Int_2[γ,β]
            @tullio Gt[m1,m2,n1,n2,γ,x] := 4/P * Gt1[m1,n1,γ,x] * Gt2[m2,n2,γ,x]
            @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
            return collect(Gᵗ)
        end
    
        G = FourierTensorTrain(GG, z=F₁.z*F₂.z)
        compress!(G; svd_trunc)
        normalize && normalize_eachmatrix!(G)
        any(any(isnan, Gᵗ) for Gᵗ in G) && @error "NaN in Fourier tensor train"
        return (G, 1.0, d₁+d₂)
    end

    TTinit = [[1/2 for _ in 1:1, _ in 1:1, y in 1:2, x in 1:2] for _ in 1:length(F[1])] |> TensorTrain
    Ginit = (FourierTensorTrain_spin(TTinit, K, Inf, P, 1/1.5K), 1.0, 0)
    G, full = cavity(zip(F,J,fill(1,length(F))) |> collect, op, Ginit)
    return G, full
end

function convolution(F::Vector{FourierMPEM1{U1}}, J::Vector{U2}, P::Float64;
    kw...) where {U1<:Number, U2<:Real}
    F2 = [[(@tullio _[a,b,c,d] := fᵗ[a,b,c] d∈1:2) |> collect for fᵗ in f] for f in F]
    FMPEM2 = FourierTensorTrain.(F2)
    convolution(FMPEM2,J,P; kw...)
end



@noinline function _compute_integral(β,Jⱼᵢ,xⱼᵗ,hᵢ,xᵢᵗ⁺¹,kᵧ, y, ::Val{scale}) where scale
    if scale<0 || β<0
        # @infiltrate scale<0
        @show β scale
        display(Base.@locals)
        @assert false  
    end
    @noinline function _compute_primitive_1(X, β, scale, Jxj, kᵧ, xp1, bb, cc, denom)
        expon = exp(im*kᵧ*X + β*xp1*(Jxj+scale*X))
        hypgeom = _₂F₁(1.0, bb, cc, -exp(2β*(Jxj+scale*X)))
        return expon / denom * hypgeom
    end
    @noinline function _compute_primitive_2(X, β, scale, Jxj)
        # @show "1" β
        a = β*scale * (1 + exp(2β * (Jxj+scale*X)))
        return (2β * (Jxj+scale*X) - log(a)) / (2β*scale)
        # return X + Jxj/scale - (log(β*scale) + log(1 + exp(2β * (Jxj+scale*X)))) / (2β*scale)
    end

    Jxj = Jⱼᵢ*xⱼᵗ + hᵢ
    if iszero(kᵧ) && xᵢᵗ⁺¹==-1
        return _compute_primitive_2(y[2], β, scale, Jxj) - _compute_primitive_2(y[1], β, scale, Jxj)
    else
        # @show "2" β
        xp1 = 1+xᵢᵗ⁺¹
        bb = (xp1 + im*kᵧ/(β*scale)) / 2
        cc = 1 + bb
        denom = im*kᵧ + β*scale*xp1
        return _compute_primitive_1(y[2], β, scale, Jxj, kᵧ, xp1, bb, cc, denom) - _compute_primitive_1(y[1], β, scale, Jxj, kᵧ, xp1, bb, cc, denom)
    end
end

function onebpiter_fourier!(bp::MPBP, i::Integer, K::Integer; P=2.0, σ=1/50, svd_trunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i][1], ϕ[i], length(ein)
    J, hᵢ, β = float.(wᵢ.J), wᵢ.h, wᵢ.β
    scale1 = dᵢ+1.0
    # @show scale1 β
    # display(scale1)


    μ_fourier = [FourierTensorTrain_spin(μ[k], K, scale1, P, σ) for k in idx.(ein)]
    dest, (conv_μ_full,) = convolution(μ_fourier, J, P; K, scale=scale1, svd_trunc)
    (conv_μ,) = unzip(dest)

    for (j, e_out) in enumerate(eout)
        conv_μ_notj = conv_μ[j]
        # jldsave("./check_convolution/m_$(src(e_out))-$(dst(e_out)) fourier.jld2"; conv_μ_notj)

        # @show "3" β scale
        # @tullio In[γ,xᵢᵗ⁺¹,xⱼᵗ] := _compute_integral(β,J[$j],potts2spin(xⱼᵗ),hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1.0,1.0], scale) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2
        @assert scale1>0
        In_ = [_compute_integral(β,J[j],potts2spin(xⱼᵗ),hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1.0,1.0], Val(scale1)) for γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2]
        In = OffsetArray(In_, -K:K, 1:2, 1:2)

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

    # @tullio In[γ,xᵢᵗ⁺¹] := _compute_integral(β,0.0,0.0,hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1.0,1.0], scale1) γ∈-K:K, xᵢᵗ⁺¹∈1:2
    @assert scale1>0
    In_ = [_compute_integral(β,0.0,0.0,hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/P, [-1.0,1.0], Val(scale1)) for γ∈-K:K, xᵢᵗ⁺¹∈1:2]
    In = OffsetArray(In_, -K:K, 1:2)
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

function (cb::CB_BP)(bp::MPBP, it::Integer, svd_trunc::SVDTrunc)
    marg_new = means(cb.f, bp)
    marg_old = cb.m[end]
    Δ = isempty(marg_new) ? NaN : maximum(maximum(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    push!(cb.Δs, Δ)
    push!(cb.m, marg_new)
    next!(cb.prog, showvalues=[(:Δ,Δ), summary_compact(svd_trunc)])
    flush(stdout)
    return Δ
end

function iterate_fourier!(bp::MPBP, K::Integer; maxiter::Integer=5, svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, σ::Real=1/50)
    for it in 1:maxiter
        # Threads.@threads for i in nodes
        for i in nodes
            onebpiter_fourier!(bp, i, K; svd_trunc)
        end
        Δ = cb(bp, it, svd_trunc)
        Δ < tol && return it, cb
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return maxiter, cb
end