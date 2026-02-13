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
    : m_scope(intervals),
      m_track_dir(track_dir)
{
}

static size_t find_first_overlap(const std::vector<GInterval> &intervals, int64_t start)
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

void PMSparseIterator::begin()
{
    m_out_intervals.clear();
    m_out_interval_ids.clear();

    if (m_scope.empty()) {
        m_isend = true;
        return;
    }

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    const size_t num_chroms = chromkey.get_num_chroms();
    std::vector<std::vector<std::pair<GInterval, uint64_t>>> scope_by_chrom(num_chroms);

    for (size_t i = 0; i < m_scope.size(); ++i) {
        const GInterval &interval = m_scope[i];
        if (interval.chromid < 0 || (size_t)interval.chromid >= num_chroms) {
            continue;
        }
        scope_by_chrom[interval.chromid].push_back({interval, i + 1});
    }

    for (size_t chromid = 0; chromid < num_chroms; ++chromid) {
        if (scope_by_chrom[chromid].empty()) {
            continue;
        }

        std::string chrom_file = GenomeTrack::find_existing_1d_filename(
            chromkey, m_track_dir, (int)chromid);
        std::string full_path = m_track_dir + "/" + chrom_file;

        if (access(full_path.c_str(), F_OK) != 0) {
            continue;
        }

        GenomeTrackSparse sparse;
        sparse.init_read(full_path.c_str(), (int)chromid);
        const std::vector<GInterval> &track_intervals = sparse.get_intervals();
        if (track_intervals.empty()) {
            continue;
        }

        for (const auto &scope_pair : scope_by_chrom[chromid]) {
            const GInterval &scope_interval = scope_pair.first;
            uint64_t scope_id = scope_pair.second;

            size_t idx = find_first_overlap(track_intervals, scope_interval.start);
            for (size_t i = idx; i < track_intervals.size(); ++i) {
                const GInterval &track_interval = track_intervals[i];
                if (track_interval.start >= scope_interval.end) {
                    break;
                }
                int64_t overlap_start = std::max(scope_interval.start, track_interval.start);
                int64_t overlap_end = std::min(scope_interval.end, track_interval.end);
                if (overlap_start < overlap_end) {
                    m_out_intervals.emplace_back((int)chromid, overlap_start, overlap_end);
                    m_out_interval_ids.push_back(scope_id);
                }
            }
        }
    }

    m_idx = 0;
    m_isend = m_out_intervals.empty();
}

void PMSparseIterator::next()
{
    if (m_isend) return;

    ++m_idx;
    if (m_idx >= m_out_intervals.size()) {
        m_isend = true;
    }
}
