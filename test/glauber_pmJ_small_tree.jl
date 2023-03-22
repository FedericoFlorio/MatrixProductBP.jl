```Glauber on a small tree for ±J ising model, comparison with exact solution```

rng = MersenneTwister(111)

T = 3

J = [0 -1  0  0;
     -1 0  1  1;
     0  1  0  0;
     0  1  0  0] .|> float

N = size(J, 1)
h = randn(rng, N)

β = 2.0
ising = Ising(J, h, β)

gl = Glauber(ising, T)

for i in 1:N
    r = 0.75
    gl.ϕ[i][1] .*= [r, 1-r]
end

bp = mpbp(gl)

@testset "Factor type" begin
    @test all(eltype(wᵢ) <: PMJGlauberFactor for wᵢ in bp.w)
end

X = draw_node_observations!(bp, N; rng)

svd_trunc = TruncThresh(0.0)
cb = CB_BP(bp; showprogress=false)
iterate!(bp; maxiter=20, svd_trunc, cb)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; p_exact)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)
Z_bp = exp(-f_bethe)

f(x,i) = 2x-3

r_bp = autocorrelations(f, bp)
r_exact = exact_autocorrelations(f, bp; p_exact)

c_bp = autocovariances(f, bp)
c_exact = exact_autocovariances(f, bp; r = r_exact)

@testset "Glauber ±J small tree" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
    @test c_bp ≈ c_exact
end

### Test against generic BP
Jvec = filter.(!iszero, eachcol(J))
w_generic = [[GenericGlauberFactor(Jvec[i], h[i], β) for wit in bp.w[i]] for i in 1:N]

bp_generic = mpbp(bp.g, w_generic, fill(2,N), T; ϕ=bp.ϕ)

svd_trunc = TruncThresh(0.0)
cb = CB_BP(bp_generic; showprogress=false)
iterate!(bp_generic; maxiter=20, svd_trunc, cb)

b_bp_generic = beliefs(bp_generic)
p_bp_generic = [[bbb[2] for bbb in bb] for bb in b_bp_generic]
f_bethe = bethe_free_energy(bp_generic)
Z_bp_generic = exp(-f_bethe)
r_bp_generic = autocorrelations(f, bp_generic)
c_bp_generic = autocovariances(f, bp_generic)

@testset "Glauber ±J small tree GenericGlauberFactor" begin
    @test Z_exact ≈ Z_bp_generic
    @test p_ex ≈ p_bp_generic
    @test r_bp_generic ≈ r_exact
    @test c_bp_generic ≈ c_exact
end

### Test with GenericFactor
w_slow = [[GenericFactor(wit) for wit in bp.w[i]] for i in 1:N]

bp_slow = mpbp(bp.g, w_slow, fill(2,N), T; ϕ=bp.ϕ)

svd_trunc = TruncThresh(0.0)
cb = CB_BP(bp_slow; showprogress=false)
iterate!(bp_slow; maxiter=20, svd_trunc, cb)

b_bp_slow = beliefs(bp_slow)
p_bp_slow = [[bbb[2] for bbb in bb] for bb in b_bp_slow]
f_bethe = bethe_free_energy(bp_slow)
Z_bp_slow = exp(-f_bethe)
r_bp_slow = autocorrelations(f, bp_slow)
c_bp_slow = autocovariances(f, bp_slow)

@testset "Glauber ±J small tree GenericFactor" begin
    @test Z_exact ≈ Z_bp_slow
    @test p_ex ≈ p_bp_slow
    @test r_bp_slow ≈ r_exact
    @test c_bp_slow ≈ c_exact
end

sms = SoftMarginSampler(bp)
sample!(sms, 10^6)

# using Plots
p_mc = [[bbb[2] for bbb in bb] for bb in marginals(sms)]

# ### Perturb slightly the Js and check that observables are unchanged

# J2 = copy(J)
# for i in 1:N
#     for j in i:N
#         if J[i,j] != 0
#             r = eps(1.0)
#             J2[i,j] += r
#             J2[j,i] += r
#         end
#     end
# end

# ising2 = Ising(J2, h, β)
# gl2 = Glauber(ising2, T)

# for i in 1:N
#     r = 0.75
#     gl2.ϕ[i][1] .*= [r, 1-r]
# end

# bp2 = mpbp(gl2)
# bp2.ϕ .= bp.ϕ

# cb = CB_BP(bp2; showprogress=false)
# iterate!(bp2; maxiter=20, svd_trunc, cb)
# b_bp2 = beliefs(bp2)
# p_bp2 = [[bbb[2] for bbb in bb] for bb in b_bp2]
# p_exact2, Z_exact2 = exact_prob(bp2)
# b_exact2 = exact_marginals(bp2; p_exact=p_exact2)
# p_ex2 = [[bbb[2] for bbb in bb] for bb in b_exact2]

# f_bethe2 = bethe_free_energy(bp2)
# Z_bp2 = exp(-f_bethe2)

# @testset "Glauber ±J small tree generic from perturbation" begin
#     @test Z_exact2 ≈ Z_bp2
#     @test p_ex2≈ p_bp2
#     @test Z_exact2 ≈ Z_exact
#     @test p_ex2≈ p_ex
# end