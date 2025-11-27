import Base.ComplexF64
function complex(x::AcbFieldElem)
    return ComplexF64(Float64(real(x)), Float64(imag(x)))
end

"""
    _fourier3int(α::Int, γ::Int; w::Float64=0.5, J::Float64=1.0, P::Float64=2.0)

Computes the integral ``∫dx F_α(x) F*_γ(Jx)``. The integrations ranges are ``[-w,+w]``. This is part of the calculation of the integral ``∫dx₁ ∫dx₂ F_α(x₁) F_β(x₂) F*_γ(J₁x₁+J₂x₂)``.
"""
function _fourier3int(α::Int, γ::Int, w::Real, J::Float64, Pscale::Float64, d₁::Integer, d₂::Integer)
    d₁₂ = d₁ + d₂
    kαγ = 2π * (d₁₂*α - d₁*J*γ) / (d₁*d₁₂*Pscale)
    iszero(kαγ) ? w : sin(kαγ*w)/kαγ
end

cavity2(data, op, init) = [reduce(op, (data[i] for i in eachindex(data) if i ≠ j); init=init) for j in eachindex(data)], reduce(op, data; init=init)

function convolution(F::Vector{<:MPEM2{U1}}, J::Vector{U2}, P::Float64, T::Integer; K::Int=(size(F[1][1],3)-1)/2, svd_trunc=TruncThresh(1e-8), normalize::Bool=true, scale::Float64=1.0, σ::Float64=1/100) where {U1<:Number, U2<:Real}

    function op((F₁, J₁, d₁), (F₂, J₂, d₂))
        K1 = (size(F₁[1],3)-1)/2 |> Int
        K2 = (size(F₂[1],3)-1)/2 |> Int
        @tullio Int_1[γ,α] := _fourier3int(α,γ,d₁*scale,J₁,P*scale,d₁,d₂) α∈-K1:K1, γ∈-K:K
        @tullio Int_2[γ,β] := _fourier3int(β,γ,d₂*scale,J₂,P*scale,d₂,d₁) β∈-K2:K2, γ∈-K:K
    
        GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
            @tullio Gt1[m1,n1,γ,x] := F₁ᵗ[m1,n1,α,x] * Int_1[γ,α]
            @tullio Gt2[m2,n2,γ,x] := F₂ᵗ[m2,n2,β,x] * Int_2[γ,β]
            @tullio Gt[m1,m2,n1,n2,γ,x] := 4/(P*scale*(d₁+d₂)) * Gt1[m1,n1,γ,x] * Gt2[m2,n2,γ,x]
            @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
            return collect(Gᵗ)
        end
    
        G = fourier_tensor_train(GG, z=F₁.z*F₂.z)
        compress!(G; svd_trunc)
        normalize && normalize_eachmatrix!(G)
        any(any(isnan, Gᵗ) for Gᵗ in G) && @error "NaN in Fourier tensor train"
        return (G, 1.0, d₁+d₂)
    end

    # ttinit = [exp(-2*π^2 * α^2 * σ^2 / (P*scale)^2) for _ in 1:1, _ in 1:1, α in -K:K, _ in 1:2]
    # ginit = [ComplexF64.(ttinit) for _ in 1:T+1]
    # ginit_tt = fourier_tensor_train(ginit)
    # normalize_eachmatrix!(ginit_tt)

    ginit = [[1.0 + 0.0*im for _ in 1:1, _ in 1:1, α in -K:K, _ in 1:2] for _ in 1:T+1] |> fourier_tensor_train
    Ginit = (ginit, 1.0, 1)

    G, full = cavity(zip(F,J,fill(1,length(F))) |> collect, op, Ginit)
    return G, full
end

function convolution(F::Vector{MPEM1{U1}}, J::Vector{U2}, P::Float64, T::Integer;
    kw...) where {U1<:Number, U2<:Real}
    F2 = [[(@tullio _[a,b,c,d] := fᵗ[a,b,c] d∈1:2) |> collect for fᵗ in f] for f in F]
    FMPEM2 = fourier_tensor_train.(F2)
    convolution(FMPEM2,J,P,T; kw...)
end


function _compute_integral(β,Jⱼᵢ,xⱼᵗ,hᵢ,xᵢᵗ⁺¹,kᵧ, yy)
    function _compute_primitive_1(X,β,Jxj,bb)
        hypgeom = 0.0 + 0.0im
        precbits = 64
        while true
            CC = AcbField(precbits)
            a_ = CC(1.0)
            b_ = CC(bb)
            c_ = CC(1+bb)
            x_ = CC(-exp(2β*(Jxj+X)))
            hyp_nemo = hypergeometric_2f1(a_, b_, c_, x_)
            
            if !isnan(Float64(real(hyp_nemo))+Float64(real(hyp_nemo)))
                max((Nemo.radius(real(hyp_nemo))), Nemo.radius(imag(hyp_nemo))) > 1e-12 && @error "Possible numerical instability in hypergeometric function ($hyp_nemo)"
                hypgeom = complex(hyp_nemo)
                break
            end
            precbits *= 2
        end

        isnan(hypgeom) && error("NaN in hypergeometric function")
        
        return hypgeom
    end
    function _compute_primitive_2(X)
        a = β * (1 + exp(2β * (Jxj+X)))
        return Jxj + X - log(a) / (2β)
    end

    Jxj = Jⱼᵢ*xⱼᵗ + hᵢ
    if iszero(kᵧ) && xᵢᵗ⁺¹==-1
        return _compute_primitive_2(yy) - _compute_primitive_2(-yy)
    else
        xp1 = 1+xᵢᵗ⁺¹
        bb = (xp1 + im*kᵧ/β) / 2
        denom = bb * 2 * β

        exp_prim_p1 = exp(im*kᵧ*yy + β*xp1*(Jxj+yy)) * _compute_primitive_1(yy,β,Jxj,bb)
        exp_prim_m1 = exp(-im*kᵧ*yy + β*xp1*(Jxj-yy)) * _compute_primitive_1(-yy,β,Jxj,bb)
        return (exp_prim_p1 - exp_prim_m1) / denom
    end
end


function _f_bp_msg_fourier(conv_μ_notj, ϕᵢ, β, J, hᵢ, Pscale, ybound, K)
    @tullio avx=false In[γ,xᵢᵗ⁺¹,xⱼᵗ] := _compute_integral(β,J,potts2spin(xⱼᵗ),hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/Pscale, ybound) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:2
    any(isnan, In) && error("NaN in integral")

    μj = map(eachindex(conv_μ_notj)) do t
        μᵗ₋ⱼ, ϕᵢᵗ = conv_μ_notj[t], ϕᵢ[t]
        @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μᵗ₋ⱼ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹,xⱼᵗ] * ϕᵢᵗ[xᵢᵗ]
        return Aᵗ
    end
    μᵀ₋ⱼ, ϕᵢᵀ = conv_μ_notj[end], ϕᵢ[end]
    @tullio μjᵀ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μᵀ₋ⱼ[m,n,0,xᵢᵗ] * Pscale * ϕᵢᵀ[xᵢᵗ] xⱼᵗ∈1:2, xᵢᵗ⁺¹∈1:2
    μj[end] = μjᵀ

    return collect.(μj) |> MPEM3 |> mpem2
end

function _f_bp_belief_fourier(conv_μ_full, ϕᵢ, β, hᵢ, Pscale, ybound, K)
    @tullio avx=false In[γ,xᵢᵗ⁺¹] := _compute_integral(β,0.0,0.0,hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/Pscale, ybound) γ∈-K:K, xᵢᵗ⁺¹∈1:2
    any(isnan, In) && error("NaN in integral")

    b = map(eachindex(conv_μ_full)) do t
        μ_fullᵗ, ϕᵢᵗ = conv_μ_full[t], ϕᵢ[t]
        @tullio Aᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μ_fullᵗ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹] * ϕᵢᵗ[xᵢᵗ] xⱼᵗ∈1:2
        return Aᵗ
    end
    μ_fullᵀ, ϕᵢᵀ = conv_μ_full[end], ϕᵢ[end]
    @tullio bT[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := μ_fullᵀ[m,n,0,xᵢᵗ] * Pscale * ϕᵢᵀ[xᵢᵗ] xⱼᵗ∈1:2, xᵢᵗ⁺¹∈1:2
    b[end] = bT

    return collect.(b) |> MPEM3 |> mpem2 |> marginalize
end


function onebpiter_fourier!(bp::MPBP, i::Integer, K::Integer; P=2.0, σ=1/50, svd_trunc=TruncThresh(1e-6), damp=0.0)
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i][1], ϕ[i], length(ein)
    J, hᵢ, β = float.(wᵢ.J), wᵢ.h, wᵢ.β
    scale = 1 + ceil(dᵢ/4) / (dᵢ+1)   # = (dᵢ + 1 + ceil(dᵢ/4)) / (dᵢ+1)

    μ_fourier = [fourier_tensor_train_spin(μ[k], K, scale, P, σ) for k in idx.(ein)]
    dest, full = convolution(μ_fourier, J, P, getT(bp); K, svd_trunc, scale, σ)
    (conv_μ, _) = unzip(dest)
    (conv_μ_full, _) = full
    

    for (j, e_out) in enumerate(eout)
        conv_μ_notj = conv_μ[j]
        μᵢⱼ = _f_bp_msg_fourier(conv_μ_notj, ϕᵢ, β, J[j], hᵢ, dᵢ*scale*P, dᵢ*P/2, K)
        compress!(μᵢⱼ; svd_trunc)
        normalize!(μᵢⱼ)
        set_msg!(bp, μᵢⱼ, idx(e_out), damp, svd_trunc)
    end

    belief = _f_bp_belief_fourier(conv_μ_full, ϕᵢ, β, hᵢ, (dᵢ+1)*scale*P, (dᵢ+1)*P/2 , K)
    compress!(belief; svd_trunc)
    normalize!(belief)
    bp.b[i] = belief

    return nothing
end

function iterate_fourier!(bp::MPBP, K::Integer; maxiter::Integer=5, svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, nodes=collect(vertices(bp.g)), shuffle_nodes::Bool=true, σ::Real=1/50, damp=0.0)
    @showprogress desc="Fourier MPBP $(svd_trunc)" for it in 1:maxiter
        Threads.@threads for i in nodes
        # for i in nodes
            onebpiter_fourier!(bp, i, K; svd_trunc, σ, damp)
        end
        Δ = cb(bp, it, svd_trunc)
        Δ < tol && return it, cb
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
        # println("Iteration $(it) completed")
    end
    return maxiter, cb
end

function onebpiter_fourier_infinite_regular!(bp::MPBP, K::Integer; P=2.0, σ=1/50, svd_trunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, ψ, μ = bp
    μ = only(μ)
    wᵢ, ϕᵢ, dᵢ  = w[1][1], ϕ[1], length(edges(g))
    J, hᵢ, β = float(wᵢ.J[1]), wᵢ.h, wᵢ.β
    scale = 1 + ceil(dᵢ/4) / (dᵢ+1)   # = (dᵢ + 1 + ceil(dᵢ/4)) / (dᵢ+1)

    μ_fourier = fourier_tensor_train_spin(μ, K, scale, P, σ)

    function op((F₁, J₁, d₁), (F₂, J₂, d₂))
        K1 = (size(F₁[1],3)-1)/2 |> Int
        K2 = (size(F₂[1],3)-1)/2 |> Int
        @tullio Int_1[γ,α] := _fourier3int(α,γ,d₁*scale,J₁,P*scale,d₁,d₂) α∈-K1:K1, γ∈-K:K
        @tullio Int_2[γ,β] := _fourier3int(β,γ,d₂*scale,J₂,P*scale,d₂,d₁) β∈-K2:K2, γ∈-K:K
    
        GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
            @tullio Gt1[m1,n1,γ,x] := F₁ᵗ[m1,n1,α,x] * Int_1[γ,α]
            @tullio Gt2[m2,n2,γ,x] := F₂ᵗ[m2,n2,β,x] * Int_2[γ,β]
            @tullio Gt[m1,m2,n1,n2,γ,x] := 4/(P*scale*(d₁+d₂)) * Gt1[m1,n1,γ,x] * Gt2[m2,n2,γ,x]
            @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
            return collect(Gᵗ)
        end
    
        G = fourier_tensor_train(GG, z=F₁.z*F₂.z)
        compress!(G; svd_trunc)
        normalize && normalize_eachmatrix!(G)
        any(any(isnan, Gᵗ) for Gᵗ in G) && @error "NaN in Fourier tensor train"
        return (G, 1.0, d₁+d₂)
    end

    ginit = [[1.0 + 0.0*im for _ in 1:1, _ in 1:1, α in -K:K, _ in 1:2] for _ in 1:T+1] |> fourier_tensor_train
    conv = (ginit, 1.0, 1)
    upd = (μ_fourier, J, 1)
    for d in 1:dᵢ-1
        conv = op(conv, upd)
    end
    conv_full = op(conv, upd)
    (conv_μ,), (conv_μ_full,) = conv, conv_full

    μᵢⱼ = _f_bp_msg_fourier(conv_μ, ϕᵢ, β, J[1], hᵢ, dᵢ*scale*P, dᵢ*P/2, K)
    compress!(μᵢⱼ; svd_trunc)
    normalize!(μᵢⱼ)
    bp.μ[1] = μᵢⱼ

    belief = _f_bp_belief_fourier(conv_μ_full, ϕᵢ, β, hᵢ, (dᵢ+1)*scale*P, (dᵢ+1)*P/2, K)
    compress!(belief; svd_trunc)
    normalize!(belief)
    bp.b[1] = belief

    return nothing
end

function iterate_fourier_infinite_regular!(bp::MPBP, K::Integer; maxiter::Integer=5, svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, nodes=collect(vertices(bp.g)), shuffle_nodes::Bool=true, σ::Real=1/50)
    for it in 1:maxiter
        onebpiter_fourier_infinite_regular!(bp, K; σ, svd_trunc)
        Δ = cb(bp, it, svd_trunc)
        Δ < tol && return it, cb
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return maxiter, cb
end

function pairbelief(μ1, μ2)
    map(eachindex(μ1)) do t
        μ1ᵗ, μ2ᵗ = μ1[t], μ2[t]
        @tullio b_pair_ᵗ[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ] := μ1ᵗ[m1,n1,xᵢᵗ,xⱼᵗ] * μ2ᵗ[m2,n2,xⱼᵗ,xᵢᵗ]
        @cast b_pairᵗ[(m1,m2),(n1,n2),xᵢᵗ,xⱼᵗ] := b_pair_ᵗ[m1,m2,n1,n2,xᵢᵗ,xⱼᵗ]
    end |> TensorTrain
end

function onebpiter_fourier_popdyn!(μ::Vector{<:TensorTrain{F,N}}, K, wᵢ, dᵢ, ϕᵢ, svd_trunc, P, T, σ) where {F,N}
    wᵢᵗ = wᵢ[1]
    J, hᵢ, β = float.(wᵢᵗ.J), wᵢᵗ.h, wᵢᵗ.β
    scale = 1 + ceil(dᵢ/4) / (dᵢ+1)   # = (dᵢ + 1 + ceil(dᵢ/4)) / (dᵢ+1)
    μold = deepcopy(μ)

    μ_fourier = TensorTrain{F,N}[fourier_tensor_train_spin(μk, K, scale, P, σ) for μk in μ]
    dest, (conv_μ_full,) = convolution(μ_fourier, J, P, T; K, svd_trunc, scale, σ)
    (conv_μ,) = unzip(dest)

    for j in eachindex(conv_μ)
        conv_μ_notj = conv_μ[j]
        μⱼ = _f_bp_msg_fourier(conv_μ_notj, ϕᵢ, β, J[j], hᵢ, dᵢ*scale*P, dᵢ*P/2, K)
        compress!(μⱼ; svd_trunc)
        normalize!(μⱼ)
        μ[j] = deepcopy(μⱼ)
    end
    belief = _f_bp_belief_fourier(conv_μ_full, ϕᵢ, β, hᵢ, (dᵢ+1)*scale*P, (dᵢ+1)*P/2 , K)
    compress!(belief; svd_trunc)
    normalize!(belief)

    b_pairs = map(eachindex(μ, μold)) do j
        μj = μ[j]
        μoldj = μold[j]
        pairbelief(μj, μoldj)
    end
    
    return μ, belief, b_pairs
end

function iterate_fourier_popdyn!(μ_pop::AbstractVector{<:TensorTrain{F,N}}, popsize, bs, bs2times, bs2vars, prob_degree, prob_J, prob_h, K::Integer, β, ϕᵢ, T; maxiter=100, svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress=true, tol=1e-10, P::Real=2.0, σ::Real=1/50) where {F<:Complex,N}
    @showprogress Threads.@threads for it in 1:maxiter
            dᵢ = rand(prob_degree)
            dᵢ > popsize && error("Sampled degree $(dᵢ) greater than population size $popsize")
            # dᵢ < 1 && continue

            indices = rand(eachindex(μ_pop), dᵢ)
            J = rand(prob_J, dᵢ)
            h = rand(prob_h)
            wᵢ = fill(GlauberFactor(J, h, β), T+1)

            μ, b, b_pairs = onebpiter_fourier_popdyn!(deepcopy(μ_pop[indices]), K, wᵢ, dᵢ, ϕᵢ, svd_trunc, P, T, σ)
            μ_pop[indices] .= μ

            push!(bs, [real.(m) for m in marginals(b)])
            push!(bs2times, [real.(m) for m in twovar_marginals(b)])
            for (j, b_pair) in enumerate(b_pairs)
                push!(bs2vars, ([real.(m) for m in twovar_marginals(b_pair)], J[j], dᵢ))
            end
    end
end