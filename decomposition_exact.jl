# decomposition_exact.jl
#
# Exact-arithmetic formalization of the possibilistic decomposition layer.
#
# The decomposition is the algebraic core of the method (synthetic_demo.py,
# feasible_interval / classify): given an ensemble of feasible models, the
# per-cell feasible interval [a_min, a_max] and the forced-sign classification.
# Unlike the forward solvers (FMM, Levenberg-Marquardt), which are inherently
# Float64-numerical, the decomposition is min / max / order comparison - it is
# exact, and its properties are provable.
#
# This module re-expresses it over exact Rational{BigInt} arithmetic and
# verifies its algebraic properties. Per the evidence discipline of
# inverse_born_methodology.md:
#
#   T1  PROVED       - classify is total, mutually exclusive, exhaustive.
#                      The proof is exhaustive: the classification depends only
#                      on the position of a_min, a_max relative to the deadband
#                      [-eps, eps]; every realizable configuration is enumerated
#                      with exact-rational witnesses, so the finite check covers
#                      the whole domain {a_min <= a_max, eps >= 0}.
#   T2  PROVED       - forced-high and forced-low are disjoint (corollary of T1).
#   T3  exact-verified - the forced sets are monotone non-increasing under
#                      ensemble enlargement (proof sketch in-line; checked
#                      exactly on random rational ensembles).
#   T4  exact-verified - forced-high/low shrink and forced-quiet grows as the
#                      deadband eps grows.
#   T5  exact-verified - forced-high equals the intersection, over models, of
#                      the per-model positive sets.
#
# Run:  julia decomposition_exact.jl
# Dependencies: Julia Base + the Random stdlib only.

using Random

@enum Label ForcedHigh ForcedLow ForcedQuiet MeasureDependent

const Q = Rational{BigInt}

# --- the decomposition, exact ------------------------------------------------

"Per-cell feasible interval [a_min, a_max] over the ensemble."
function feasible_interval(ensemble::Vector{Vector{T}}) where {T}
    ncell = length(first(ensemble))
    amin = [minimum(m[c] for m in ensemble) for c in 1:ncell]
    amax = [maximum(m[c] for m in ensemble) for c in 1:ncell]
    return amin, amax
end

"Forced-sign classification of one cell from its feasible interval."
function classify(amin::T, amax::T, eps::T) where {T}
    if amin > eps
        return ForcedHigh
    elseif amax < -eps
        return ForcedLow
    elseif amin >= -eps && amax <= eps
        return ForcedQuiet
    else
        return MeasureDependent
    end
end

"Decompose an ensemble into the per-cell forced-sign labels."
function decompose(ensemble::Vector{Vector{T}}, eps::T) where {T}
    amin, amax = feasible_interval(ensemble)
    return [classify(amin[c], amax[c], eps) for c in 1:length(amin)]
end

# --- random exact rationals (for the non-exhaustive verifications) -----------

randq(rng) = big(rand(rng, -30:30)) // big(rand(rng, 1:12))
randens(rng, nmodel, ncell) = [[randq(rng) for _ in 1:ncell] for _ in 1:nmodel]

# --- T1 / T2 : exhaustive proof ----------------------------------------------

function check_config(amin::Q, amax::Q, eps::Q)
    @assert amin <= amax && eps >= 0
    p_high  = amin > eps                       # ForcedHigh guard
    p_low   = amax < -eps                      # ForcedLow guard
    p_quiet = amin >= -eps && amax <= eps       # ForcedQuiet guard
    # exclusivity: at most one forced guard is true
    sum((p_high, p_low, p_quiet)) <= 1 || return false
    # well-definedness: the branch order in classify does not matter
    expected = p_high  ? ForcedHigh :
               p_low   ? ForcedLow :
               p_quiet ? ForcedQuiet : MeasureDependent
    return classify(amin, amax, eps) == expected
end

"T1/T2 - exhaustive proof over every realizable interval/deadband position."
function verify_T1()
    ok, configs = true, 0
    # eps > 0  (witness eps = 1): five position classes for a value -
    # below / at / inside / at / above the deadband edges.
    five = [Q(-2), Q(-1), Q(0), Q(1), Q(2)]
    for i in 1:5, j in i:5
        configs += 1
        ok &= check_config(five[i], five[j], Q(1))
    end
    # eps == 0  (deadband collapses to {0}): three position classes.
    three = [Q(-1), Q(0), Q(1)]
    for i in 1:3, j in i:3
        configs += 1
        ok &= check_config(three[i], three[j], Q(0))
    end
    return ok, configs
end

# --- T3 : monotone in the ensemble -------------------------------------------
# Proof sketch: E subset of E' => min over E' <= min over E and max over E' >=
# max over E. So a_min can only fall and a_max only rise as the ensemble grows;
# hence {a_min > eps}, {a_max < -eps}, {a_min >= -eps and a_max <= eps} can each
# only shrink. forced structure is monotone non-increasing; measure-dependent
# grows. Verified exactly on random rational ensembles.

function verify_T3(rng; trials = 500)
    eps = Q(1, 10)
    for _ in 1:trials
        E  = randens(rng, 3, 6)
        Ep = vcat(E, randens(rng, 2, 6))            # E' is a superset of E
        lE, lEp = decompose(E, eps), decompose(Ep, eps)
        for c in 1:6
            for L in (ForcedHigh, ForcedLow, ForcedQuiet)
                lEp[c] == L && lE[c] != L && return false
            end
        end
    end
    return true
end

# --- T4 : monotone in the deadband -------------------------------------------
# Proof sketch: for eps' > eps, a_min > eps' implies a_min > eps, so forced-high
# shrinks; likewise forced-low; while a_min >= -eps' and a_max <= eps' are both
# easier, so forced-quiet grows.

function verify_T4(rng; trials = 500)
    eps, eps2 = Q(1, 10), Q(1, 3)                   # eps2 > eps
    for _ in 1:trials
        E = randens(rng, 5, 6)
        lo, hi = decompose(E, eps), decompose(E, eps2)
        for c in 1:6
            hi[c] == ForcedHigh && lo[c] != ForcedHigh && return false
            hi[c] == ForcedLow  && lo[c] != ForcedLow  && return false
            lo[c] == ForcedQuiet && hi[c] != ForcedQuiet && return false
        end
    end
    return true
end

# --- T5 : intersection characterization --------------------------------------
# Proof: a_min is the min over models; a_min > eps holds iff every model's value
# exceeds eps. So forced-high is exactly the intersection, over models, of the
# per-model sets {cell : model(cell) > eps}.

function verify_T5(rng; trials = 500)
    eps = Q(1, 10)
    for _ in 1:trials
        E = randens(rng, 4, 6)
        viaInterval = decompose(E, eps)
        for c in 1:6
            viaIntersection = all(m[c] > eps for m in E)
            (viaInterval[c] == ForcedHigh) == viaIntersection || return false
        end
    end
    return true
end

# --- end-to-end exact demonstration ------------------------------------------

function demo()
    ensemble = Vector{Q}[
        [Q(3,4),  Q(-1,2), Q(1,10),  Q(-3,5),  Q(0)    ],
        [Q(7,8),  Q(-2,5), Q(-1,4),  Q(-7,10), Q(1,20) ],
        [Q(2,3),  Q(-3,4), Q(2,5),   Q(-1,2),  Q(-1,15)],
        [Q(9,10), Q(-1,3), Q(-1,3),  Q(-4,5),  Q(1,12) ],
    ]
    eps = Q(1, 5)
    amin, amax = feasible_interval(ensemble)
    labels = decompose(ensemble, eps)
    println("Exact end-to-end decomposition (eps = $(eps)):")
    for c in 1:length(labels)
        println("  cell $c:  interval [$(amin[c]), $(amax[c])]   ->  $(labels[c])")
    end
end

# --- run ---------------------------------------------------------------------

function main()
    println("="^70)
    println("Possibilistic decomposition - exact-arithmetic formalization")
    println("Rational{BigInt}; Julia Base + Random only")
    println("="^70)

    ok1, n1 = verify_T1()
    println("\nT1/T2  classify total / exclusive / exhaustive")
    println("       $(ok1 ? "PROVED" : "FAILED") - exhaustive over $(n1) "
            * "realizable configurations (exact rational witnesses)")

    rng = MersenneTwister(20260517)
    ok3 = verify_T3(rng)
    println("T3     forced sets monotone non-increasing in the ensemble")
    println("       $(ok3 ? "exact-verified" : "FAILED") - 500 random rational ensembles")

    ok4 = verify_T4(rng)
    println("T4     forced-high/low shrink, forced-quiet grows, as eps grows")
    println("       $(ok4 ? "exact-verified" : "FAILED") - 500 random rational ensembles")

    ok5 = verify_T5(rng)
    println("T5     forced-high = intersection of per-model positive sets")
    println("       $(ok5 ? "exact-verified" : "FAILED") - 500 random rational ensembles")

    println()
    demo()

    allok = ok1 && ok3 && ok4 && ok5
    println("\n", "="^70)
    println(allok ? "ALL CHECKS PASS" : "SOME CHECKS FAILED")
    println("="^70)
    allok || exit(1)
end

main()
