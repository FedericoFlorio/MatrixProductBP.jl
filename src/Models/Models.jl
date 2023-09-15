module Models

import MatrixProductBP: exact_prob, getT, nstates, mpbp, compress!,
    kron2, f_bp, f_bp_dummy_neighbor, onebpiter_dummy_neighbor,
    beliefs, beliefs_tu, marginals, pair_belief, pair_beliefs,
    marginalize, cavity, onebpiter!, check_ψs, _compose,
    RecursiveBPFactor, nstates, prob_y, prob_xy, prob_yy, prob_y_partial,
    prob_y_dummy
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedBiDiGraph, AbstractIndexedDiGraph, ne, nv, 
    outedges, idx, src, dst, inedges, neighbors, edges, vertices, IndexedEdge
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals, Symmetric
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

export 
    Ising, Glauber, energy,
    HomogeneousGlauberFactor, GenericGlauberFactor, PMJGlauberFactor, mpbp,
    equilibrium_magnetization, equilibrium_observables, RandomRegular, ErdosRenyi, CB_Pop,
    SIS, SISFactor, SIRS, SIRSFactor, SUSCEPTIBLE, INFECTED, RECOVERED,
    kl_marginals, l1_marginals, roc, auc

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")
include("glauber/equilibrium.jl")

include("epidemics/sis.jl")
include("epidemics/sis_bp.jl")
include("epidemics/inference.jl")
include("epidemics/sirs.jl")
include("epidemics/sirs_bp.jl")

end # end module
