# function onebpiter_infinite_graph(A::MPEM2, k::Integer, pᵢ⁰, wᵢ::Vector{U}, 
#         ϕᵢ, ψₙᵢ;
#         svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {U<:SimpleBPFactor}
    
#     B = f_bp_glauber(fill(A, k), pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, 1)
#     C = mpem2(B)
#     A_new = sweep_RtoL!(C; svd_trunc)
#     normalize_eachmatrix!(A_new)
#     A_new
# end


# function iterate_glauber_infinite_graph(T::Integer, k::Integer, pᵢ⁰, wᵢ;
#         ϕᵢ = fill(ones(q_glauber), T),
#         ψₙᵢ = fill(fill(ones(q_glauber,q_glauber), T), k),
#         svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
#         showprogress=true)

#     A = mpem2(q_glauber, T)
#     Δs = fill(NaN, maxiter)
#     m = firstvar_marginal(A)
#     dt = showprogress ? 0.1 : Inf
#     prog = Progress(maxiter; dt, desc="Iterating BP on infinite graph")
#     for it in 1:maxiter
#         A = onebpiter_glauber_infinite_graph(A, k, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ; 
#             svd_trunc)
#         m_new = firstvar_marginal(A)
#         Δ = maximum(abs, bb_new[1] - bb[1] for (bb_new, bb) in zip(m_new, m))
#         Δs[it] = Δ
#         Δ < tol && return A, it, Δs
#         m, m_new = m_new, m
#         rounded_Δ = round(Δ, digits=ceil(Int,abs(log(tol))))
#         next!(prog, showvalues=[(:iter, "$it/$maxiter"), (:Δ,"$rounded_Δ/$tol")])
#     end
#     A, maxiter, Δs
# end



# function observables_glauber_infinite_graph(A::MPEM2, k::Integer, m⁰::Real;
#         β::Real=1.0, J::Real=1.0, h::Real=0.0,
#         svd_trunc::SVDTrunc=TruncThresh(1e-6),
#         showprogress=true)
#     pᵢ⁰ = [(1+m⁰)/2, (1-m⁰)/2]
#     T = length(A) - 1
#     wᵢ = fill(HomogeneousGlauberFactor(fill(J, k), h, β), T)
#     r, m = autocorrelation_magnetization_glauber_infinite(A, k, pᵢ⁰, wᵢ;
#         svd_trunc, showprogress)
#     c = MatrixProductBP.covariance(r, m)
#     m, r, c
# end 

# function onebpiter_dummy_glauber_infinite_graph(A::MPEM2, k::Integer, pᵢ⁰, 
#         wᵢ, ϕᵢ, ψₙᵢ; svd_trunc::SVDTrunc=TruncThresh(1e-6))

#     B = f_bp_dummy_neighbor_glauber(fill(A, k), pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ)
#     C = mpem2(B)
#     A_new = sweep_RtoL!(C; svd_trunc)
#     normalize_eachmatrix!(A_new)
#     A_new
# end