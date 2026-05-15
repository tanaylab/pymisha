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
#include <string>
#include <vector>

#include "GInterval2D.h"
#include "PMGenomeTrack2D.h"
#include "pymisha.h"

class PMTrackExpression2DVars {
public:
    enum AggFunc { AGG_AREA, AGG_WEIGHTED_SUM, AGG_MIN, AGG_MAX, AGG_AVG };

    struct TrackVar2D {
        std::string                     name;           // track name
        std::string                     var_name;       // Python-safe name
        std::unique_ptr<PMGenomeTrack2D> track;
        AggFunc                         func;
        PMPY                            py_var;
        std::vector<double>             cpp_values;
        double                         *values{nullptr};
    };

    PMTrackExpression2DVars();
    ~PMTrackExpression2DVars();

    // Register a single track variable with an explicit agg func.
    // Used by the test binding pm_test_2d_scanner. Throws TGLException on
    // unknown track / unsupported track type / unknown agg func.
    void add_var(const std::string &track_name, const std::string &func_name);

    // Allocate per-var value buffers of length `size` (and Python ndarrays
    // if `use_python`). Mirrors PMTrackExpressionVars::define_py_vars.
    void define_py_vars(unsigned size, PMPY &ldict, bool use_python);

    // For each var, fill values[start_idx .. start_idx+count-1] by
    // grouping `intervals[start_idx..start_idx+count-1]` by chrom-pair,
    // doing one query_stats_batch per group, and applying the var's agg
    // func to the result. `band` may be nullptr.
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
};

#endif /* PMTRACKEXPRESSION2DVARS_H_ */
