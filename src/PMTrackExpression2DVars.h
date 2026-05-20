/*
 * PMTrackExpression2DVars.h
 *
 * Manages 2D-track variables for the 2D scanner. Parallel to
 * PMTrackExpressionVars but specialised for GInterval2D inputs and 2D
 * track aggregation funcs.
 *
 * Differences from the 1D version:
 *   - Each TrackVar2D carries an explicit aggregation func (area /
 *     weighted.sum / min / max / avg). 1D vars don't need this - the
 *     reduction is implicit in how each track type is read.
 *   - set_vars_batch consumes a contiguous run of buffered 2D intervals
 *     and groups them by chrom-pair before issuing one query_stats_batch
 *     per group. The 1D analog calls set_vars row-by-row because the
 *     1D readers are stateful per-chromosome.
 *   - No virtual tracks, no sequence-based vars, no shift parameters.
 *     Those are deferred to Session 4.
 */

#ifndef PMTRACKEXPRESSION2DVARS_H_
#define PMTRACKEXPRESSION2DVARS_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "GInterval2D.h"
#include "PMGenomeTrack2D.h"
#include "pymisha.h"

class PMTrackExpression2DVars {
public:
    enum AggFunc {
        AGG_AREA, AGG_WEIGHTED_SUM, AGG_MIN, AGG_MAX, AGG_AVG,
        // Object-level funcs: resolved by querying intersecting objects per cell.
        AGG_EXISTS,   // 1 if any object intersects, else 0
        AGG_SIZE,     // count of intersecting objects
        AGG_FIRST,    // value of the first object (by query-iteration order), NaN if none
        AGG_LAST,     // value of the last object, NaN if none
        AGG_SAMPLE,   // value of a randomly-sampled object, NaN if none
    };

    struct TrackVar2D {
        std::string                     name;           // track name
        std::string                     var_name;       // Python-safe name
        std::unique_ptr<PMGenomeTrack2D> track;
        AggFunc                         func;
        // Per-var 2D iterator shifts (applied to each query rect before
        // querying the underlying track, matching R's vtrack shift semantics).
        int64_t                         sshift1{0};
        int64_t                         eshift1{0};
        int64_t                         sshift2{0};
        int64_t                         eshift2{0};
        PMPY                            py_var;
        std::vector<double>             cpp_values;
        double                         *values{nullptr};
    };

    PMTrackExpression2DVars();
    ~PMTrackExpression2DVars();

    // Reset all registered vars, releasing all track and buffer resources.
    // After clear() the object is in the same state as a fresh default-constructed
    // instance. Called by PMTrackExpr2DScanner::run() and run_single_var() at
    // the top of each call to prevent var accumulation across reuse.
    void clear() { m_vars.clear(); }

    // Register a single track variable with an explicit agg func and optional
    // per-var 2D shifts (defaults to zero = no shift). Throws TGLException on
    // unknown track / unsupported track type / unknown agg func.
    void add_var(const std::string &track_name, const std::string &func_name,
                 int64_t sshift1 = 0, int64_t eshift1 = 0,
                 int64_t sshift2 = 0, int64_t eshift2 = 0);

    // Allocate per-var value buffers of length `size` (and Python ndarrays
    // if `use_python`). Mirrors PMTrackExpressionVars::define_py_vars.
    void define_py_vars(unsigned size, PMPY &ldict, bool use_python);

    // For each var, fill values[start_idx .. start_idx+count-1] by
    // grouping `intervals[start_idx..start_idx+count-1]` by chrom-pair,
    // doing one query_stats_batch per group for stats-based funcs (area/
    // weighted.sum/min/max/avg), or one query_objects call per cell for
    // object-level funcs (exists/size/first/last/sample). `band` may be nullptr.
    void set_vars_batch(const std::vector<GInterval2D> &intervals,
                        unsigned start_idx, unsigned count,
                        const quadtree::DiagonalBand *band);

    // Pad the trailing slots [start_idx, end_idx) of every var's value
    // buffer with NaN (used at end-of-iterator).
    void pad_tail_with_nan(unsigned start_idx, unsigned end_idx);

    // Number of registered vars.
    unsigned num_vars() const { return static_cast<unsigned>(m_vars.size()); }

    // Read-only access for the scanner.
    const TrackVar2D &var(unsigned i) const { return m_vars[i]; }

    // Convert an agg-func string to the AggFunc enum. Throws on unknown.
    static AggFunc parse_func(const std::string &name);

    // Convert the AggFunc enum back to a name (for diagnostics).
    static const char *func_name(AggFunc f);

private:
    std::vector<TrackVar2D> m_vars;

    // Apply `func` to a quadtree::BatchQueryStats slice (n entries),
    // writing `values[indices[j]] = applied(j)` for j in [0, n).
    // Slots whose occupied_area == 0 are left untouched (caller must
    // pre-fill them with NaN).
    void apply_agg(AggFunc func,
                   const quadtree::BatchQueryStats &batch,
                   const std::vector<unsigned> &indices,
                   double *values);

    // Apply an object-level func (exists/size/first/last/sample) by calling
    // query_objects for each index in `indices`. The var's track must already
    // have the correct chrom-pair open. `band` may be nullptr.
    void apply_agg_objects(TrackVar2D &var,
                           AggFunc func,
                           const std::vector<GInterval2D> &intervals,
                           const std::vector<unsigned> &indices,
                           const quadtree::DiagonalBand *band);

    // True if func is one of the five object-level funcs.
    static bool is_object_func(AggFunc f) {
        return f == AGG_EXISTS || f == AGG_SIZE ||
               f == AGG_FIRST || f == AGG_LAST || f == AGG_SAMPLE;
    }

    // Per-object-var RNG (seeded once in constructor, used for AGG_SAMPLE).
    std::mt19937_64 m_rng;
};

#endif /* PMTRACKEXPRESSION2DVARS_H_ */
