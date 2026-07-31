# This file is a part of Julia. License is MIT: https://julialang.org/license

### Reductions of structured matrices
#
# Reductions of banded matrices can avoid repeatedly reducing over structural zeros
# if the operation treats them as neutral elements.
#
# Each structured matrix defines a `_mapreduce_bands(f, op, A, init, inds, red)`
# decomposition that reduces each band segment covered by the rectangular index chunk
# `inds` with `red(f, op, band, init, segment)` and combines the results with `op`.
# That decomposition is exposed to Base's reduction machinery through two of its
# extension points:
#   * `Base.mapreduce_kernel`, the reduction leaf, which is only ever called on chunks
#     of at most `Base.pairwise_blocksize` indices, and
#   * `Base.mapreduce_pairwise`, which intercepts a whole (sub)reduction *before* Base
#     recursively splits it into O(length(inds)/blocksize) leaf chunks. Reducing the
#     bands directly keeps the whole reduction O(bandwidth × n), and using
#     `Base.mapreduce_pairwise` on each band segment preserves the pairwise
#     reassociation (and its accuracy) for long bands.


# The index range within the band of offset `k` (indexed such that element `e` of the
# band is at `A[e, e+k]` for `k ≥ 0`, or at `A[e-k, e]` for `k < 0`) that is covered by
# the index rectangle `(is, js)`. May be empty.
_bandinds(is, js, k) = k >= 0 ?
    (max(first(is), first(js)-k):min(last(is), last(js)-k)) :
    (max(first(is)+k, first(js)):min(last(is)+k, last(js)))

function _mapreduce_bands(f, op, A::Diagonal, init, inds::CartesianIndices{2}, red::R) where {R}
    is, js = inds.indices
    ds = _bandinds(is, js, 0)
    isempty(ds) && return Base._mapreduce_start(f, op, A, init, diagzero(A, first(inds)))
    return red(f, op, A.diag, init, ds)
end

function _mapreduce_bands(f, op, A::Bidiagonal, init, inds::CartesianIndices{2}, red::R) where {R}
    length(inds) == 1 && return Base._mapreduce_start(f, op, A, init, A[first(inds)])
    is, js = inds.indices
    ds = _bandinds(is, js, 0)
    r = if isempty(ds)
        Base._mapreduce_start(f, op, A, init, diagzero(A, first(inds)))
    else
        red(f, op, A.dv, init, ds)
    end
    es = _bandinds(is, js, A.uplo == 'U' ? 1 : -1)
    isempty(es) || (r = op(r, red(f, op, A.ev, init, es)))
    return r
end

function _mapreduce_bands(f, op, A::Tridiagonal, init, inds::CartesianIndices{2}, red::R) where {R}
    length(inds) == 1 && return Base._mapreduce_start(f, op, A, init, A[first(inds)])
    is, js = inds.indices
    ds = _bandinds(is, js, 0)
    r = if isempty(ds)
        # a chunk of more than one element that misses the diagonal contains at
        # least one structural zero; include a single representative
        Base._mapreduce_start(f, op, A, init, diagzero(A, first(inds)))
    else
        red(f, op, A.d, init, ds)
    end
    us = _bandinds(is, js, 1)
    ls = _bandinds(is, js, -1)
    isempty(us) || (r = op(r, red(f, op, A.du, init, us)))
    isempty(ls) || (r = op(r, red(f, op, A.dl, init, ls)))
    return r
end

function _mapreduce_bands(f, op, A::SymTridiagonal, init, inds::CartesianIndices{2}, red::R) where {R}
    length(inds) == 1 && return Base._mapreduce_start(f, op, A, init, A[first(inds)])
    is, js = inds.indices
    ds = _bandinds(is, js, 0)
    r = if isempty(ds)
        Base._mapreduce_start(f, op, A, init, diagzero(A, first(inds)))
    else
        symmetric(red(f, op, A.dv, init, ds), :U)
    end
    us = _bandinds(is, js, 1)
    ls = _bandinds(is, js, -1)
    if !isempty(us)
        eu = red(f, op, A.ev, init, us)
        r = op(r, eu)
        # both off-diagonals cover the same `ev` segment for (near-)diagonal chunks;
        # reuse the reduction rather than performing it twice
        if us == ls
            return op(r, transpose(eu))
        end
    end
    isempty(ls) || (r = op(r, transpose(red(f, op, A.ev, init, ls))))
    return r
end

for MT in (:Diagonal, :Bidiagonal, :Tridiagonal, :SymTridiagonal)
    @eval begin
        Base.mapreduce_kernel(f::typeof(identity), op::Union{typeof(+), typeof(Base.add_sum)}, A::$MT, init, inds::CartesianIndices{2}) =
            _mapreduce_bands(f, op, A, init, inds, Base.mapreduce_kernel)
        Base.mapreduce_pairwise(f::typeof(identity), op::Union{typeof(+), typeof(Base.add_sum)}, A::$MT, init, inds::CartesianIndices{2}) =
            _mapreduce_bands(f, op, A, init, inds, Base.mapreduce_pairwise)
    end
end

# For idempotent operations, duplicated elements cannot affect the result, so these
# reductions support arbitrary `f` and represent the *entire* zero region by a single
# `f(diagzero)` whenever the reduced chunk extends beyond the bands. Note that this
# set must not include non-idempotent operations like `+` (for which the above methods
# instead rely on the structural zeros being skippable identity elements, and are thus
# limited to `f = identity`).
const IdempotentReduceOps = Union{typeof(min), typeof(max), typeof(&), typeof(|), typeof(Base.and_all), typeof(Base.or_any)}

_bandoffsets(::Diagonal) = (0,)
_bandoffsets(A::Bidiagonal) = A.uplo == 'U' ? (0, 1) : (-1, 0)
_bandoffsets(::Union{Tridiagonal, SymTridiagonal}) = (-1, 0, 1)

function _mapreduce_bands_withzeros(f, op, A, init, inds::CartesianIndices{2}, red::R) where {R}
    r = _mapreduce_bands(f, op, A, init, inds, red)
    is, js = inds.indices
    nband = 0
    for k in _bandoffsets(A)
        nband += length(_bandinds(is, js, k))
    end
    if nband < length(inds)
        # the chunk extends beyond the bands, so it contains at least one structural
        # zero; with `Number` eltypes every `diagzero` is the same `zero(T)`, so the
        # corner of maximal |i-j| serves as the representative
        i0, j0 = last(js) - first(is) >= last(is) - first(js) ?
            (first(is), last(js)) : (last(is), first(js))
        r = op(r, f(diagzero(A, i0, j0)))
    end
    return r
end

# These methods are restricted to `Number` eltypes (as the old whole-array
# `minimum`/`maximum` methods for Diagonal were): with matrix-valued elements,
# `diagzero`'s shape varies by position, so a single `f(diagzero)` cannot stand in
# for every structural zero, and for SymTridiagonal `f` would additionally need to
# commute with the `symmetric`/`transpose` remapping of its band storage.
for MT in (:(Diagonal{<:Number}), :(Bidiagonal{<:Number}), :(Tridiagonal{<:Number}), :(SymTridiagonal{<:Number}))
    @eval begin
        Base.mapreduce_kernel(f, op::IdempotentReduceOps, A::$MT, init, inds::CartesianIndices{2}) =
            _mapreduce_bands_withzeros(f, op, A, init, inds, Base.mapreduce_kernel)
        Base.mapreduce_pairwise(f, op::IdempotentReduceOps, A::$MT, init, inds::CartesianIndices{2}) =
            _mapreduce_bands_withzeros(f, op, A, init, inds, Base.mapreduce_pairwise)
    end
end
