/*
 * PMTrackExpressionIterator.h
 *
 * Iterator classes for track expression evaluation in pymisha
 */

#ifndef PMTRACKEXPRESSIONITERATOR_H_
#define PMTRACKEXPRESSIONITERATOR_H_

#include <vector>
#include <memory>

#include "GInterval.h"
#include "PMDb.h"
#include "GenomeTrackSparse.h"

// Base class for all track expression iterators
class PMTrackExpressionIterator {
public:
    virtual ~PMTrackExpressionIterator() {}

    virtual void begin() = 0;
    virtual void next() = 0;
    virtual bool isend() const = 0;

    virtual const GInterval &last_interval() const = 0;

    // Estimated size (0 if unknown)
    virtual uint64_t size() const { return 0; }

    // Current index (for progress reporting)
    virtual uint64_t idx() const { return 0; }

    // Original input interval index (1-based, for intervalID column)
    virtual uint64_t original_interval_idx() const { return 0; }

    // Is this a 1D iterator?
    virtual bool is_1d() const { return true; }
};


// Iterator that divides intervals into fixed-size bins
class PMFixedBinIterator : public PMTrackExpressionIterator {
public:
    PMFixedBinIterator(const std::vector<GInterval> &intervals, int64_t bin_size,
                       bool interval_relative = false);
    virtual ~PMFixedBinIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval &last_interval() const override { return m_cur_interval; }

    virtual uint64_t size() const override { return m_size; }
    virtual uint64_t idx() const override { return m_idx; }
    virtual uint64_t original_interval_idx() const override { return m_cur_interv_idx + 1; }  // 1-based

    int64_t get_bin_size() const { return m_bin_size; }

private:
    std::vector<GInterval> m_intervals;
    int64_t m_bin_size;
    bool m_interval_relative;
    size_t m_cur_interv_idx;
    GInterval m_cur_interval;
    bool m_isend;
    uint64_t m_size;
    uint64_t m_idx;
    int64_t m_cur_bin;
    int64_t m_end_bin;
    GInterval m_last_scope_interval;

    void calc_size();
};


// Iterator that returns intervals as-is (one interval per iteration)
class PMIntervalsIterator : public PMTrackExpressionIterator {
public:
    PMIntervalsIterator(const std::vector<GInterval> &intervals);
    virtual ~PMIntervalsIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval &last_interval() const override { return m_intervals[m_cur_idx]; }

    virtual uint64_t size() const override { return m_intervals.size(); }
    virtual uint64_t idx() const override { return m_cur_idx; }
    virtual uint64_t original_interval_idx() const override { return m_cur_idx + 1; }  // 1-based

private:
    std::vector<GInterval> m_intervals;
    size_t m_cur_idx;
    bool m_isend;
};


// Iterator that returns sparse track intervals overlapping the scope.
// Streams overlaps lazily: loads one chromosome at a time and computes
// overlaps on-the-fly during next(), avoiding upfront materialization.
class PMSparseIterator : public PMTrackExpressionIterator {
public:
    PMSparseIterator(const std::vector<GInterval> &intervals, const std::string &track_dir);
    virtual ~PMSparseIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval &last_interval() const override { return m_cur_overlap; }

    virtual uint64_t size() const override { return 0; }  // unknown for streaming
    virtual uint64_t idx() const override { return m_total_emitted; }
    virtual uint64_t original_interval_idx() const override { return m_cur_scope_id; }

private:
    struct ScopeEntry {
        GInterval interval;
        uint64_t scope_id;  // 1-based original interval index
    };

    std::string m_track_dir;
    // Scope intervals grouped by chromosome; only chroms with scope intervals are stored
    std::vector<int> m_chrom_order;  // chromids in order
    std::vector<std::vector<ScopeEntry>> m_scope_by_chrom;

    // Current chromosome state
    size_t m_cur_chrom_order_idx{0};
    GenomeTrackSparse m_cur_track;
    const std::vector<GInterval> *m_cur_track_intervals{nullptr};

    // Current iteration state within a chromosome
    size_t m_cur_scope_idx{0};   // index into scope intervals for current chrom
    size_t m_cur_track_idx{0};   // index into track intervals for current overlap scan

    // Current output
    GInterval m_cur_overlap;
    uint64_t m_cur_scope_id{0};
    uint64_t m_total_emitted{0};
    bool m_isend{true};

    // Load sparse track data for the given chrom_order_idx
    bool load_chrom(size_t chrom_order_idx);

    // Advance to the next valid overlap; return true if found
    bool find_next_overlap();

    // Binary search for first track interval that could overlap pos
    size_t find_first_overlap(const std::vector<GInterval> &intervals, int64_t pos) const;
};

#endif /* PMTRACKEXPRESSIONITERATOR_H_ */
