/*
 * PMTrackExpressionIterator.cpp
 *
 * Iterator classes for track expression evaluation in pymisha
 */

#include "PMTrackExpressionIterator.h"
#include "TGLException.h"

#include <algorithm>
#include <cmath>
#include <unistd.h>

// ========================= PMFixedBinIterator =========================

PMFixedBinIterator::PMFixedBinIterator(const std::vector<GInterval> &intervals, int64_t bin_size,
                                       bool interval_relative)
    : m_intervals(intervals),
      m_bin_size(bin_size),
      m_interval_relative(interval_relative),
      m_cur_interv_idx(0),
      m_isend(true),
      m_size(0),
      m_idx(0)
{
    if (bin_size <= 0) {
        TGLError("Invalid bin size: %ld", bin_size);
    }
    calc_size();
}

void PMFixedBinIterator::calc_size()
{
    m_size = 0;
    for (const auto &interval : m_intervals) {
        int64_t start_bin = 0;
        int64_t end_bin = 0;

        if (m_interval_relative) {
            int64_t interval_len = interval.end - interval.start;
            end_bin = (int64_t)ceil(interval_len / (double)m_bin_size);
        } else {
            start_bin = (int64_t)(interval.start / (double)m_bin_size);
            end_bin = (int64_t)ceil(interval.end / (double)m_bin_size);
        }

        if (end_bin > start_bin) {
            m_size += (uint64_t)(end_bin - start_bin);
        }
    }
}

void PMFixedBinIterator::begin()
{
    m_cur_interv_idx = 0;
    m_idx = 0;
    m_cur_bin = m_end_bin = -1;

    if (m_intervals.empty()) {
        m_isend = true;
        return;
    }

    m_isend = false;
    m_last_scope_interval = m_intervals[0];

    // Initialize to the first bin without advancing the progress index.
    if (m_cur_bin != m_end_bin)
        m_cur_bin++;

    if (m_cur_bin == m_end_bin) {
        if (m_interval_relative) {
            m_cur_bin = 0;
            m_end_bin = (int64_t)ceil(
                (m_last_scope_interval.end - m_last_scope_interval.start) / (double)m_bin_size
            );
        } else {
            m_cur_bin = (int64_t)(m_last_scope_interval.start / (double)m_bin_size);
            m_end_bin = (int64_t)ceil(m_last_scope_interval.end / (double)m_bin_size);
        }
        m_cur_interval.chromid = m_last_scope_interval.chromid;
    }

    int64_t coord = m_interval_relative
        ? m_last_scope_interval.start + m_cur_bin * m_bin_size
        : m_cur_bin * m_bin_size;
    m_cur_interval.start = std::max(coord, m_last_scope_interval.start);
    m_cur_interval.end = std::min(coord + m_bin_size, m_last_scope_interval.end);
}

void PMFixedBinIterator::next()
{
    if (m_isend) return;

    m_idx++;

    if (m_cur_bin != m_end_bin)
        m_cur_bin++;

    if (m_cur_bin == m_end_bin) {
        if (m_cur_bin >= 0) {
            m_cur_interv_idx++;
            if (m_cur_interv_idx < m_intervals.size()) {
                m_last_scope_interval = m_intervals[m_cur_interv_idx];
            }
        }

        if (m_cur_interv_idx >= m_intervals.size()) {
            m_isend = true;
            return;
        }

        if (m_interval_relative) {
            m_cur_bin = 0;
            m_end_bin = (int64_t)ceil(
                (m_last_scope_interval.end - m_last_scope_interval.start) / (double)m_bin_size
            );
        } else {
            m_cur_bin = (int64_t)(m_last_scope_interval.start / (double)m_bin_size);
            m_end_bin = (int64_t)ceil(m_last_scope_interval.end / (double)m_bin_size);
        }
        m_cur_interval.chromid = m_last_scope_interval.chromid;
    }

    int64_t coord = m_interval_relative
        ? m_last_scope_interval.start + m_cur_bin * m_bin_size
        : m_cur_bin * m_bin_size;
    m_cur_interval.start = std::max(coord, m_last_scope_interval.start);
    m_cur_interval.end = std::min(coord + m_bin_size, m_last_scope_interval.end);
}


// ========================= PMIntervalsIterator =========================

PMIntervalsIterator::PMIntervalsIterator(const std::vector<GInterval> &intervals)
    : m_intervals(intervals),
      m_cur_idx(0),
      m_isend(true)
{
}

void PMIntervalsIterator::begin()
{
    m_cur_idx = 0;
    m_isend = m_intervals.empty();
}

void PMIntervalsIterator::next()
{
    if (m_isend) return;

    m_cur_idx++;
    if (m_cur_idx >= m_intervals.size()) {
        m_isend = true;
    }
}


// ========================= PMSparseIterator =========================

PMSparseIterator::PMSparseIterator(const std::vector<GInterval> &intervals, const std::string &track_dir)
    : m_track_dir(track_dir)
{
    // Organize scope intervals by chromosome
    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    const size_t num_chroms = chromkey.get_num_chroms();

    // Temporary per-chrom grouping
    std::vector<std::vector<ScopeEntry>> tmp(num_chroms);
    for (size_t i = 0; i < intervals.size(); ++i) {
        const GInterval &interval = intervals[i];
        if (interval.chromid >= 0 && (size_t)interval.chromid < num_chroms) {
            tmp[interval.chromid].push_back({interval, i + 1});
        }
    }

    // Compact: only keep chroms with scope intervals
    for (size_t chromid = 0; chromid < num_chroms; ++chromid) {
        if (!tmp[chromid].empty()) {
            m_chrom_order.push_back((int)chromid);
            m_scope_by_chrom.push_back(std::move(tmp[chromid]));
        }
    }
}

size_t PMSparseIterator::find_first_overlap(const std::vector<GInterval> &intervals, int64_t start) const
{
    size_t left = 0;
    size_t right = intervals.size();

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (intervals[mid].end <= start) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

bool PMSparseIterator::load_chrom(size_t chrom_order_idx)
{
    while (chrom_order_idx < m_chrom_order.size()) {
        int chromid = m_chrom_order[chrom_order_idx];
        std::string chrom_file = GenomeTrack::find_existing_1d_filename(
            g_pmdb->chromkey(), m_track_dir, chromid);
        std::string full_path = m_track_dir + "/" + chrom_file;

        if (access(full_path.c_str(), F_OK) == 0) {
            m_cur_track.init_read(full_path.c_str(), chromid);
            m_cur_track_intervals = &m_cur_track.get_intervals();
            if (!m_cur_track_intervals->empty()) {
                m_cur_chrom_order_idx = chrom_order_idx;
                m_cur_scope_idx = 0;
                m_cur_track_idx = 0;
                return true;
            }
        }
        ++chrom_order_idx;
    }
    return false;
}

bool PMSparseIterator::find_next_overlap()
{
    while (m_cur_chrom_order_idx < m_chrom_order.size()) {
        int chromid = m_chrom_order[m_cur_chrom_order_idx];
        const auto &scope_entries = m_scope_by_chrom[m_cur_chrom_order_idx];
        const auto &track_ivs = *m_cur_track_intervals;

        while (m_cur_scope_idx < scope_entries.size()) {
            const GInterval &scope_iv = scope_entries[m_cur_scope_idx].interval;
            uint64_t scope_id = scope_entries[m_cur_scope_idx].scope_id;

            // Position track cursor when starting a new scope interval
            if (m_cur_track_idx == 0) {
                m_cur_track_idx = find_first_overlap(track_ivs, scope_iv.start);
            }

            while (m_cur_track_idx < track_ivs.size()) {
                const GInterval &track_iv = track_ivs[m_cur_track_idx];
                if (track_iv.start >= scope_iv.end) {
                    break;
                }

                int64_t overlap_start = std::max(scope_iv.start, track_iv.start);
                int64_t overlap_end = std::min(scope_iv.end, track_iv.end);
                if (overlap_start < overlap_end) {
                    m_cur_overlap = GInterval(chromid, overlap_start, overlap_end);
                    m_cur_scope_id = scope_id;
                    ++m_cur_track_idx;  // advance for next call
                    return true;
                }
                ++m_cur_track_idx;
            }

            // Exhausted track intervals for this scope interval
            ++m_cur_scope_idx;
            m_cur_track_idx = 0;  // reset for next scope interval
        }

        // Exhausted all scope intervals for this chromosome, try next
        size_t next_chrom = m_cur_chrom_order_idx + 1;
        if (!load_chrom(next_chrom)) {
            return false;
        }
    }
    return false;
}

void PMSparseIterator::begin()
{
    m_total_emitted = 0;
    m_cur_chrom_order_idx = 0;

    if (m_chrom_order.empty()) {
        m_isend = true;
        return;
    }

    // Load first chromosome with data
    if (!load_chrom(0)) {
        m_isend = true;
        return;
    }

    // Find first overlap
    if (find_next_overlap()) {
        m_isend = false;
    } else {
        m_isend = true;
    }
}

void PMSparseIterator::next()
{
    if (m_isend) return;

    ++m_total_emitted;
    if (!find_next_overlap()) {
        m_isend = true;
    }
}
