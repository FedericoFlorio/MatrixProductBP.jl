const SUSCEPTIBLE = 1
const INFECTIOUS = 2

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

# compute derivatives of the log-likelihood with respect to infection rates of incoming edges
function der_λ(bp::MPBP{G,F}, i::Integer, ::Type{U}; svd_trunc::SVDTrunc=TruncThresh(1e-6), 
logpriorder::Function=(x)->0.0, probys::Tuple=()) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    M2 = eltype(μ)
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    qi = nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    function opt((B1ᵗ,d1),(B2ᵗ,d2),Pyy,wᵢᵗ)
        @tullio avx=false Pyy[y,y1,y2,xᵢ] = prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2) 
        @tullio BB[m1,m2,n1,n2,y,xᵢ] := Pyy[y,y1,y2,xᵢ] * B1ᵗ[m1,n1,y1,xᵢ] * B2ᵗ[m2,n2,y2,xᵢ]
        @cast _[(m1,m2),(n1,n2),y,xᵢ] := BB[m1,m2,n1,n2,y,xᵢ]
    end

    if isempty(probys)
        C, full, B = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)
        Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
        bᵢ = Bᵢ |> mpem2 |> marginalize
        zᵢ = exp(Logarithmic, normalize!(bᵢ))
    else
        (C, zᵢ, B) = probys
    end

    λder = zeros(length(ein))
    for j in eachindex(ein)
        μⱼᵢ, ψᵢⱼ = μin[j], ψout[j]
        (Bj,_) = B[j]
        Cj = C[j]

        q = length(ϕᵢ[1])
        X = [zeros(size(b,1), size(b,2), q, 1, q) for b in Bj]
        Xs = [zeros(size(b,1), size(b,2), q, 1, q) for b in Bj]
        for t in 1:T
            wᵢᵗ, μⱼᵢᵗ, ψᵢⱼᵗ = wᵢ[t], μⱼᵢ[t], ψᵢⱼ[t]
            Bⱼᵗ,Cⱼᵗ = Bj[t], Cj[t]
            Pyy = zeros(nstates(wᵢᵗ,dᵢ), size(Bⱼᵗ,3), size(Cⱼᵗ,3), size(Bⱼᵗ,4))

            Aᵗ = opt((Bⱼᵗ,1),(Cⱼᵗ,dᵢ-1),Pyy,wᵢᵗ)
            @tullio Bⱼᵗ[m,n,yⱼ,xᵢ] = ((yⱼ==INFECTIOUS)-(yⱼ==SUSCEPTIBLE)) * ψᵢⱼᵗ[xᵢ,$INFECTIOUS] * μⱼᵢᵗ[m,n,$INFECTIOUS,xᵢ]
            Asᵗ = opt((Bⱼᵗ,1),(Cⱼᵗ,dᵢ-1),Pyy,wᵢᵗ)

            W = zeros(q,q,1,size(Aᵗ,3))
            @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob_y(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,yᵗ,dᵢ)*ϕᵢ[$t][xᵢᵗ]
            @tullio Xᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * Aᵗ[m,n,yᵗ,xᵢᵗ]
            @tullio Xsᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * Asᵗ[m,n,yᵗ,xᵢᵗ]
            X[t], Xs[t] = Xᵗ, Xsᵗ
        end

        Cⱼᵀ,Bⱼᵀ,wᵢᵀ,ψᵢⱼᵀ,μⱼᵢᵀ = Cj[end], Bj[end],wᵢ[end], ψᵢⱼ[end], μⱼᵢ[end]
        Pyy = zeros(nstates(wᵢᵀ,dᵢ), size(Bⱼᵀ,3), size(Cⱼᵀ,3), size(Bⱼᵀ,4))

        Aᵀ = opt((Bⱼᵀ,1),(Cⱼᵀ,dᵢ-1),Pyy,wᵢᵀ)
        @tullio Bⱼᵀ[m,n,yⱼ,xᵢ] = ((yⱼ==INFECTIOUS)-(yⱼ==SUSCEPTIBLE)) * ψᵢⱼᵀ[xᵢ,$INFECTIOUS] * μⱼᵢᵀ[m,n,$INFECTIOUS,xᵢ]
        Asᵀ = opt((Bⱼᵀ,1),(Cⱼᵀ,dᵢ-1),Pyy,wᵢᵀ)

        @tullio Xᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] := Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ] xⱼᵀ in 1:1, xᵢᵀ⁺¹ in 1:qi
        @tullio Xsᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] := Asᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ] xⱼᵀ in 1:1, xᵢᵀ⁺¹ in 1:qi
        any(any(isnan, x) for x in X) && @error "NaN in tensor train"
        any(any(isnan, x) for x in Xs) && @error "NaN in tensor train"
        X[end], Xs[end] = Xᵀ, Xsᵀ

        X0 = MPEM3(X)
        (L,zL) = acc_L(X0)
        (R,zR) = acc_R(X0)

        der = Logarithmic(0.0)
        for s in 2:T
            LL = L[s-1]
            RR = R[s+1]
            XXs = Xs[s]
            @tullio M[i,l] := LL[i,j,x] * XXs[j,k,x,xⱼ,y] * RR[k,l,y]
            Z = only(M) * zL[s-1] * zR[s+1]
            der += Z
        end
        Xs1 = Xs[1]
        R2 = R[2]
        Lend = L[end-1]
        Xsend = Xs[end]
        @tullio M[j,l] := Xs1[j,k,x,xⱼ,y] * R2[k,l,y]  # s=1
        der += only(M) * zR[2]
        @tullio M[i,k] := Lend[i,j,x] * Xsend[j,k,x,xⱼ,y]  # s=T+1
        der += only(M) * zL[end-1]

        λder[j] = der / Cj.z / zᵢ + logpriorder(wᵢ[1].λ[j])
    end

    return λder
end

# computes derivatives of the log-likelihood with respect to recovery rate
function der_ρ(bp::MPBP{G,F}, i::Integer, ::Type{U}; svd_trunc::SVDTrunc=TruncThresh(1e-6), 
    logpriorder::Function=(x)->0.0, probys::Tuple=()) where {G<:AbstractIndexedDiGraph, F<:Real, U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    qi = nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    if isempty(probys)
        C, full, B, = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)
        Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
        bᵢ = Bᵢ |> mpem2 |> marginalize
        zᵢ = exp(Logarithmic, normalize!(bᵢ))
    else
        (full, zᵢ) = probys
    end

    q = length(ϕᵢ[1])
    X = [zeros(size(a,1), size(a,2), q, 1, q) for a in full]
    Xs = [zeros(size(a,1), size(a,2), q, 1, q) for a in full]
    for t in 1:T
        fullᵗ,Xᵗ,Xsᵗ = full[t], X[t], Xs[t]
        W = zeros(q,q,1,size(fullᵗ,3))

        @tullio avx=false W[xᵢᵗ⁺¹,INFECTIOUS,xⱼᵗ,yᵗ] = ((xᵢᵗ⁺¹==SUSCEPTIBLE)-(xᵢᵗ⁺¹==INFECTIOUS)) * ϕᵢ[$t][INFECTIOUS]
        @tullio Xsᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * fullᵗ[m,n,yᵗ,xᵢᵗ]

        @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob_y(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,yᵗ,1) * ϕᵢ[$t][xᵢᵗ]
        @tullio Xᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * fullᵗ[m,n,yᵗ,xᵢᵗ]
    end
    fullᵀ,Xᵀ,Xsᵀ = full[end], X[end], Xs[end]
    @tullio Xᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = fullᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, x) for x in X) && @error "NaN in tensor train"
    any(any(isnan, x) for x in Xs) && @error "NaN in tensor train"

    X0 = MPEM3(X)
    (L,zL) = acc_L(X0)
    (R,zR) = acc_R(X0)

    ρder = Logarithmic(0.0)
    for s in 2:T
        LL = L[s-1]
        RR = R[s+1]
        XXs = Xs[s]
        @tullio M[i,l] := LL[i,j,x] * XXs[j,k,x,xⱼ,y] * RR[k,l,y]
        Z = only(M) * zL[s-1] * zR[s+1]
        ρder += Z
    end
    Xs1 = Xs[1]
    R2 = R[2]
    @tullio M[j,l] := Xs1[j,k,x,xⱼ,y] * R2[k,l,y]  # s=1
    ρder += only(M) * zR[2]

    return float(ρder / full.z / zᵢ)
end

function derivatives(bp::MPBP{G,F}, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0) where {G<:AbstractIndexedDiGraph,F<:Real}
    @unpack g, w, ϕ, ψ, μ = bp
    T = getT(bp)
    ein, eout = inedges(g,i), outedges(g,i)
    wᵢ, ϕᵢ, dᵢ, qi = w[i], ϕ[i], length(ein), nstates(bp,i)
    μin = μ[ein.|>idx]
    ψout = ψ[eout.|>idx]

    C, full, B = compute_prob_ys(wᵢ, qi, μin, ψout, T, svd_trunc)
    Bᵢ = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bᵢ = Bᵢ |> mpem2 |> marginalize
    zᵢ = exp(Logarithmic, normalize!(bᵢ))

    λder = der_λ(bp, i, eltype(bp.w[i]); svd_trunc, logpriorder, probys=(C,zᵢ,B))
    
    ρder = der_ρ(bp, i, eltype(bp.w[i]); svd_trunc, logpriorder, probys=(full,zᵢ))

    return λder, ρder, zᵢ
end

# performs one step of Gradient Ascent for parameters of node i (i.e. infection rates of incoming edges and recovery rate)
function stepga!(bp::MPBP{G,F}, i::Integer, λstep::F=1e-2, ρstep::F=1e-2; method::Integer=1, svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0, progress::Float64=Inf) where {G<:AbstractIndexedDiGraph,F<:Real}
    @unpack g, w = bp
    ein = inedges(g,i)
    wᵢ, dᵢ = w[i], length(ein)

    λder, ρder, zᵢ = derivatives(bp, i; svd_trunc, logpriorder)
    
    if method==1
        Threads.@threads for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*λder[j]
            end
        wᵢ[t].ρ += ρstep*ρder
        end

    elseif method==2
        Threads.@threads for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] *= (1 + λstep*sign(λder[j]))
            end
            wᵢ[t].ρ *= (1 + ρstep*sign(ρder))
        end

    elseif method==21
        Threads.@threads for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] *= (1 + λstep*sign(λder[j])*5^(progress<.2))
            end
            wᵢ[t].ρ *= (1 + ρstep*sign(ρder)*5^(progress<.2))
        end

    elseif method==3
        Threads.@threads for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*sign(λder[j])
            end
            wᵢ[t].ρ += ρstep*sign(ρder)
        end

    elseif method==31
        Threads.@threads for t in eachindex(wᵢ)
            for j in 1:dᵢ
                wᵢ[t].λ[j] += λstep*sign(λder[j])*5^(progress<.25)
            end
            wᵢ[t].ρ += ρstep*sign(ρder)*5^(progress<.25)
        end

    elseif method==4
        Threads.@threads for t in eachindex(wᵢ)
            wᵢ[t].λ .+= λstep .* tanh.(0.5.*λder)
            wᵢ[t].ρ += ρstep * tanh(0.5*ρder)
        end

    elseif method==41
        Threads.@threads for t in eachindex(wᵢ)
            wᵢ[t].λ .+= λstep .* tanh.(0.001.*λder) * 3^(progress<.25)
            wᵢ[t].ρ += ρstep * tanh(0.001*ρder) * 3^(progress<.25)
        end

    else
        @error "Invalid method"
    end

    for t in eachindex(wᵢ)
        for j in 1:dᵢ
            wᵢ[t].λ[j] = clamp(wᵢ[1].λ[j], 1e-9, 1-1e-9)
        end
        wᵢ[t].ρ = clamp(wᵢ[1].ρ, 1e-9, 1-1e-9)
    end

    return zᵢ
end

# what follows is the implementation of a callback to save parameters during the iterations and print info
struct PARAMS{T}
    λ :: Vector{Vector{T}}
    ρ :: Vector{T}
    function PARAMS(λ::Vector{Vector{T}},ρ::Vector{T}) where {T<:Real}
        new{T}(λ,ρ)
    end
end

function save_data(bp::MPBP; sites=vertices(bp.g))
    @unpack w = bp
    λ = [similar(w[i][1].λ) for i in sites]
    ρ = zeros(length(sites))

    for i in sites
        wᵢ = w[i]
        λ[i] = [λⱼᵢ for λⱼᵢ in wᵢ[1].λ]
        ρ[i] = wᵢ[1].ρ
    end

    return PARAMS(λ, ρ)
end

struct CB_INF{TP<:ProgressUnknown, F, T}
    prog :: TP
    data :: T
    Δs   :: Vector{Float64}
    logz :: Vector{Float64}
    f    :: F

    function CB_INF(bp::MPBP; showprogress::Bool=true, f::F=save_data, info="") where F
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running Gradient Ascent: iter", dt=dt)
        TP = typeof(prog)
        data = [save_data(bp)]
        T = typeof(data)
        Δs = zeros(0)
        logz = zeros(0)
        new{TP,F,T}(prog, data, Δs, logz, f)
    end
end

function (cb_inf::CB_INF)(bp::MPBP, logz::Float64, it::Integer, svd_trunc::SVDTrunc)
    err(old, new) = abs.(new .- old ./ old)
    data_new = save_data(bp)
    @unpack λ, ρ = data_new
    λnew, ρnew = λ, ρ
    data_old = cb_inf.data[end]
    @unpack λ, ρ = data_old
    λold, ρold = λ, ρ

    Δ = max(maximum(err(ρold,ρnew)), maximum(maximum(err(λold[i],λnew[i])) for i in eachindex(λold)))
    push!(cb_inf.Δs, Δ)
    push!(cb_inf.data, data_new)
    push!(cb_inf.logz,logz)
    next!(cb_inf.prog, showvalues=[(:Δ,Δ), summary_compact(svd_trunc)])
    flush(stdout)
    return Δ
end

function inference_parameters!(bp::MPBP; method::Integer=1, maxiter::Integer=5, λstep=1e-2, ρstep=1e-2, svd_trunc::SVDTrunc=TruncThresh(1e-6), logpriorder::Function=(x)->0.0, showprogress=true, tol=1e-10, nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, cb_inf=CB_INF(bp;showprogress), verbose=true)

    for iter in 1:maxiter
        # Threads.@threads for i in nodes
        for i in nodes
            onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc)#, damp=0.0)
        end

        logz = 0.0
        Threads.@threads for i in nodes
            logz += log(stepga!(bp, i, λstep, ρstep; method, svd_trunc, logpriorder, progress=iter/maxiter))
        end

        Δ = cb_inf(bp,logz,iter,svd_trunc)
        Δ < tol && return iter, cb_inf
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end

    return maxiter, cb_inf
end