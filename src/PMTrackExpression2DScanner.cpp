/*
 * PMTrackExpression2DScanner.cpp
 *
 * Drives the 2D iterator and fills a per-var value buffer. The pattern
 * mirrors PMTrackExprScanner::eval_next, but with no Python expression
 * compilation. The buffer fill size is taken from the global pymisha
 * config (m_eval_buf_size); we cap it at the input length so very
 * small inputs don't waste a 1000-row allocation.
 */

#include "PMTrackExpression2DScanner.h"

#include <algorithm>
#include <cmath>

#include "PMTrackExpression2DIteratorPolicy.h"
#include "pymisha.h"

// g_pymisha (declared in pymisha.h) is a PyMisha * holding global config
// including the eval_buf_size used to size batch-fill loops.

PMTrackExpr2DScanner::PMTrackExpr2DScanner() = default;

PMTrackExpr2DScanner::~PMTrackExpr2DScanner() = default;

void PMTrackExpr2DScanner::run_single_var(
    const std::string &track_name,
    const std::string &func_name,
    const std::vector<GInterval2D> &intervals,
    const quadtree::DiagonalBand *band)
{
    // Reset all state from any prior run.
    m_vars.clear();
    m_num_intervals = intervals.size();

    // Register a single var. add_var throws on bad track / bad func.
    m_vars.add_var(track_name, func_name);

    // Pick a batch size: min(global buffer config, total intervals). For
    // very small inputs we don't want a 1000-slot allocation just to
    // immediately pad it with NaN.
    int config_buf = g_pymisha ? g_pymisha->eval_buf_size() : 1000;
    if (config_buf <= 0) config_buf = 1000;
    m_eval_buf_limit = static_cast<unsigned>(
        std::min<size_t>(static_cast<size_t>(config_buf),
                         std::max<size_t>(1, m_num_intervals)));

    // Allocate per-var value buffers sized to the *full* input length -
    // unlike the 1D scanner we keep the entire result in memory because
    // the only consumer (test binding / pm_extract_2d) wants the whole
    // array at once. Using PMPY (with `ldict = nullptr`) is OK; we just
    // get a numpy array that the binding will return.
    PMPY ldict;  // null - we don't expose vars to Python yet
    m_vars.define_py_vars(static_cast<unsigned>(m_num_intervals), ldict, true);

    // Drive the iterator.
    m_itr = std::make_unique<PMTrackExpressionIntervals2DIterator>(intervals);
    m_itr->begin();

    // Process in batches. The buffer-fill loop pulls up to m_eval_buf_limit
    // intervals at a time, then calls set_vars_batch on that slice.
    unsigned cursor = 0;
    while (!m_itr->isend()) {
        unsigned batch_start = cursor;
        unsigned batch_count = 0;
        while (!m_itr->isend() && batch_count < m_eval_buf_limit) {
            m_itr->next();
            // m_itr->idx() advances on every next(); we don't need it
            // here because intervals are passed by reference into
            // set_vars_batch and indexed by absolute position.
            ++cursor;
            ++batch_count;
        }
        m_vars.set_vars_batch(intervals, batch_start, batch_count, band);
    }
}

const double *PMTrackExpr2DScanner::values() const
{
    if (m_vars.num_vars() == 0) return nullptr;
    return m_vars.var(0).values;
}

void PMTrackExpr2DScanner::run(
    const PMTrackExpression2DIteratorPolicy &policy,
    const std::vector<VarSpec> &vars,
    const std::vector<GInterval2D> &scope,
    const quadtree::DiagonalBand *band)
{
    // Reset all state from any prior run.
    m_vars.clear();
    m_emitted_intervals.clear();
    m_emitted = 0;
    m_num_intervals = 0;

    // Register all vars with their per-var shifts (throws on invalid track / func).
    for (const auto &vs : vars) {
        m_vars.add_var(vs.name, vs.func,
                       vs.sshift1, vs.eshift1,
                       vs.sshift2, vs.eshift2);
    }

    // Build the polymorphic iterator from the policy and walk it.
    // Walk pattern: read last_interval() then call next().
    // This is correct for both Intervals2D (begin() does not prime) and
    // FixedRect (constructor calls begin()->next() so first cell is
    // already live on construction; the outer while(!isend) guard is
    // what allows us to read before advancing).
    // run_single_var uses a different advance-first loop; both patterns
    // are correct for their respective iterator contracts.
    auto itr = policy.make_iterator(scope, band);
    itr->begin();
    while (!itr->isend()) {
        // NOTE: the full emitted set is materialized up-front. For FixedRect at HiC
        // scale (~10^8 cells) this is several GB (sizeof(GInterval2D) ~= 32 bytes).
        // A streaming variant will be needed in a later release; for release-1
        // workloads (test DB) this is acceptable.
        m_emitted_intervals.push_back(itr->last_interval());
        itr->next();
    }

    m_emitted = m_emitted_intervals.size();
    m_num_intervals = m_emitted;

    if (m_emitted == 0) return;

    PMPY ldict;
    m_vars.define_py_vars(static_cast<unsigned>(m_emitted), ldict, true);

    int config_buf = g_pymisha ? g_pymisha->eval_buf_size() : 1000;
    if (config_buf <= 0) config_buf = 1000;
    unsigned batch = static_cast<unsigned>(
        std::min<size_t>(static_cast<size_t>(config_buf), m_emitted));

    for (unsigned cursor = 0; cursor < m_emitted; cursor += batch) {
        unsigned count = std::min(batch,
                                  static_cast<unsigned>(m_emitted - cursor));
        m_vars.set_vars_batch(m_emitted_intervals, cursor, count, band);
    }
}

const double *PMTrackExpr2DScanner::values_for_var(unsigned i) const
{
    if (i >= m_vars.num_vars()) return nullptr;
    return m_vars.var(i).values;
}
