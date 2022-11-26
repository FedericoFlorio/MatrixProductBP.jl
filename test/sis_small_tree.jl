q = q_sis
T = 3

A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.2
ρ = 0.1
γ = 0.1

sis = SIS(g, λ, ρ, T; γ)
bp = mpbp(sis)
rng = MersenneTwister(111)
X, _ = onesample(bp; rng)
draw_node_observations!(bp.ϕ, X, N, last_time=true; rng)

svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

b_bp = beliefs(bp; svd_trunc)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; p_exact)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp; svd_trunc)
Z_bp = exp(-f_bethe)

r_bp = autocorrelations(bp; svd_trunc)
r_exact = exact_autocorrelations(bp)

c_bp = autocovariances(bp; svd_trunc)
c_exact = exact_autocovariances(bp)

@testset "SIS small tree" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
    @test c_bp ≈ c_exact
end

# observe everything and check that the free energy corresponds to the prior of the sample `X`
draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
reset_messages!(bp)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)
f_bethe = bethe_free_energy(bp)
logl_bp = - f_bethe
logp, logl = logprior_loglikelihood(bp, X)

@testset "SIS small tree - observe everything" begin
    @test logl_bp ≈ logp
end

