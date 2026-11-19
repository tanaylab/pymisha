/*
 * PMIntervalsVtrack.h
 *
 * A virtual track whose source is a DataFrame of intervals, evaluated as R misha's
 * TrackExpressionVars evaluates one: the interval functions (distance, distance.center,
 * distance.edge, coverage, neighbor.count) as an Interv_var, the other functions as a Value_var
 * (the source's value column read as an in-memory track). The source is read once; eval() then
 * takes one iterator interval at a time, after the vtrack's shifts.
 *
 * Used by the track expression scanner (PMTrackExpressionVars) and by pm_vtrack_compute.
 */

#ifndef PMINTERVALSVTRACK_H_
#define PMINTERVALSVTRACK_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pymisha.h"
#include "GInterval.h"
#include "GenomeTrackInMemory.h"
#include "SegmentFinder.h"

class PMIntervalsVtrack {
public:
    static bool is_interval_func(const std::string &func) {
        return func == "distance" || func == "distance.center" || func == "distance.edge" ||
               func == "coverage" || func == "neighbor.count";
    }

    // spec: the vtrack's spec dict, whose "src" is the source as _df2pymisha gives it;
    // func: the vtrack's function, lower case
    PMIntervalsVtrack(PyObject *spec, const std::string &func);
    PMIntervalsVtrack(const PMIntervalsVtrack &) = delete;
    PMIntervalsVtrack &operator=(const PMIntervalsVtrack &) = delete;

    // The value over interval (the iterator interval after the shifts, start < end)
    double eval(const GInterval &interval);

    // The in-memory track of a value-based source, and the percentile of "quantile"
    GenomeTrackInMemory &track() { return m_track; }
    double percentile() const { return m_percentile; }

private:
    enum Kind { DISTANCE, DISTANCE_CENTER, DISTANCE_EDGE, COVERAGE, NEIGHBOR_COUNT, VALUE };

    Kind        m_kind;
    std::string m_func;
    double      m_param;       // params (distance margin, neighbor.count distance, percentile)
    double      m_percentile;

    // VALUE
    GenomeTrackInMemory m_track;

    // DISTANCE*, COVERAGE: the source sorted by start (COVERAGE: overlapping intervals unified)
    std::vector<GInterval> m_sintervs;
    double m_dist_margin{0};

    // DISTANCE*: nearest-neighbor index of one chromosome's source, built when first queried
    SegmentFinder<GInterval> m_finder;
    SegmentFinder<GInterval>::NNIterator m_nn{&m_finder};
    int m_finder_chromid{-1};

    // COVERAGE: lower_bound position of the last query, scanned forward while queries advance
    size_t  m_scan_idx{0};
    int64_t m_scan_last_start{0};
    int     m_scan_last_chrom{-1};

    // NEIGHBOR_COUNT: per chromosome, the starts and the ends of the source intervals expanded by
    // the distance, each sorted. An expanded interval overlaps [s, e) unless it starts at e or
    // later or ends at s or earlier, and every one that ends by s starts before e: the count is
    // #(start < e) - #(end <= s).
    std::vector<std::vector<int64_t>> m_nc_starts;
    std::vector<std::vector<int64_t>> m_nc_ends;

    void ensure_finder(int chromid);
    size_t lower_bound_start(const GInterval &interval);
};

#endif /* PMINTERVALSVTRACK_H_ */
