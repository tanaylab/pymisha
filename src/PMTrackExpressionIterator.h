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


// Iterator that returns sparse track intervals overlapping the scope
class PMSparseIterator : public PMTrackExpressionIterator {
public:
    PMSparseIterator(const std::vector<GInterval> &intervals, const std::string &track_dir);
    virtual ~PMSparseIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval &last_interval() const override { return m_out_intervals[m_idx]; }

    virtual uint64_t size() const override { return m_out_intervals.size(); }
    virtual uint64_t idx() const override { return m_idx; }
    virtual uint64_t original_interval_idx() const override { return m_out_interval_ids[m_idx]; }

private:
    std::vector<GInterval> m_scope;
    std::string m_track_dir;
    std::vector<GInterval> m_out_intervals;
    std::vector<uint64_t> m_out_interval_ids;
    size_t m_idx{0};
    bool m_isend{true};
};

#endif /* PMTRACKEXPRESSIONITERATOR_H_ */
