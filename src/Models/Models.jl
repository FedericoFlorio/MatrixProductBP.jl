module Models

import MatrixProductBP: exact_prob, getT, nstates, mpbp, compress!,
    f_bp, f_bp_dummy_neighbor, onebpiter_dummy_neighbor,
    beliefs, beliefs_tu, marginals, pair_belief, pair_beliefs,
    marginalize, cavity, onebpiter!, check_ψs, _compose,
    RecursiveBPFactor, nstates, prob_y, prob_xy, prob_yy, prob_y0, prob_y_partial,
    prob_y_dummy, periodic_mpbp, mpbp_stationary
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedDiGraph, IndexedBiDiGraph, AbstractIndexedDiGraph, ne, nv, 
    outedges, idx, src, dst, inedges, neighbors, edges, vertices, IndexedEdge
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals, Symmetric, SparseMatrixCSC, sparse
import TensorCast: @reduce, @cast, TensorCast 
import ProgressMeter: Progress, next!, ProgressUnknown
import LogExpFunctions: xlogx, xlogy
import Statistics: mean, std
import Measurements: Measurement, ±
import LoopVectorization
import Tullio: @tullio
import Unzip: unzip
import Distributions: rand, Poisson, Distribution, Dirac, MixtureModel
import Random: GLOBAL_RNG, shuffle!
import Lazy: @forward
import LinearAlgebra: issymmetric
import HypergeometricFunctions: _₂F₁

export
    Ising, Glauber, energy,
    GlauberFactor, HomogeneousGlauberFactor, GenericGlauberFactor, PMJGlauberFactor, IntegerGlauberFactor, mpbp, mpbp_stationary,
    equilibrium_magnetization, equilibrium_observables, RandomRegular, ErdosRenyi, CB_Pop,
    SIS, SISFactor, SIRS, SIRSFactor, SIS_heterogeneous, SIS_heterogeneousFactor, SUSCEPTIBLE, INFECTIOUS, RECOVERED,
    kl_marginals, l1_marginals, roc, auc,
    potts2spin, spin2potts

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")
include("glauber/equilibrium.jl")

include("epidemics/sis.jl")
include("epidemics/sis_bp.jl")
include("epidemics/inference.jl")
include("epidemics/sirs.jl")
include("epidemics/sirs_bp.jl")
include("epidemics/sis_heterogeneous.jl")
include("epidemics/sis_heterogeneous_bp.jl")

end # end module
