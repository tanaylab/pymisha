/*
 * GenomeTrackSparse.h
 *
 * Minimal sparse track reader for pymisha (ported from misha).
 */

#ifndef GENOMETRACKSPARSE_H_
#define GENOMETRACKSPARSE_H_

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "GenomeTrack1D.h"
#include "GInterval.h"

// !!!!!!!!! IN CASE OF ERROR THIS CLASS THROWS TGLException  !!!!!!!!!!!!!!!!

class GenomeTrackSparse : public GenomeTrack1D {
public:
    GenomeTrackSparse();

    void read_interval(const GInterval &interval) override;
    double last_max_pos() const override;
    double last_min_pos() const override;

    void init_read(const char *filename, int chromid);
    void init_write(const char *filename, int chromid);

    void write_next_interval(const GInterval &interval, float val);

    const std::vector<GInterval> &get_intervals();
    const std::vector<float> &get_vals();

protected:
    static const int RECORD_SIZE;

    std::vector<GInterval> m_intervals;
    std::vector<float> m_vals;
    bool m_loaded;
    int64_t m_num_records;
    size_t m_cur_idx;
    double m_last_min_pos;

    // State for indexed "smart handle"
    std::string m_dat_path;
    std::string m_dat_mode;
    bool m_dat_open{false};

    void read_file_into_mem();
    void calc_vals(const GInterval &interval);
    bool check_first_overlap(size_t idx, const GInterval &interval) const;

    void read_header_at_current_pos_(BufferedFile &bf);

    static constexpr size_t kSparseRecBytes = sizeof(int64_t) * 2 + sizeof(float);
};

inline bool GenomeTrackSparse::check_first_overlap(size_t idx, const GInterval &interval) const
{
    if (idx >= m_intervals.size())
        return false;
    const GInterval &cur = m_intervals[idx];
    if (cur.do_overlap(interval) && (idx == 0 || !m_intervals[idx - 1].do_overlap(interval)))
        return true;
    return false;
}

inline void GenomeTrackSparse::calc_vals(const GInterval &interval)
{
    // Fast path: when only basic reducers (avg/sum/min/max/nearest) are needed
    // (no position tracking, stddev, quantile, sample, exists, first/last, size)
    const bool basic_only =
        !m_functions[MIN_POS] && !m_functions[MAX_POS] &&
        !m_functions[STDDEV] && !m_use_quantile &&
        !m_functions[EXISTS] && !m_functions[FIRST] && !m_functions[FIRST_POS] &&
        !m_functions[LAST] && !m_functions[LAST_POS] &&
        !m_functions[SAMPLE] && !m_functions[SAMPLE_POS] && !m_functions[SIZE];

    if (basic_only) {
        float num_vs = 0;
        double sum = 0;
        float mn = std::numeric_limits<float>::max();
        float mx = -std::numeric_limits<float>::max();
        for (size_t i = m_cur_idx; i < m_intervals.size(); ++i) {
            const GInterval &cur = m_intervals[i];
            if (!cur.do_overlap(interval))
                break;
            float v = m_vals[i];
            if (!std::isnan(v)) {
                sum += v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                ++num_vs;
            }
        }
        if (num_vs > 0) {
            m_last_sum = sum;
            m_last_avg = m_last_nearest = sum / num_vs;
            m_last_min = mn;
            m_last_max = mx;
        } else {
            m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_sum = std::numeric_limits<float>::quiet_NaN();
        }
        return;
    }

    // Generic path: full function bookkeeping
    float num_vs = 0;
    double mean_square_sum = 0;

    std::vector<float> all_values;
    std::vector<double> all_positions;
    if (m_functions[SAMPLE] || m_functions[SAMPLE_POS])
        all_values.reserve(100);
    if (m_functions[SAMPLE_POS])
        all_positions.reserve(100);

    m_last_sum = 0;
    m_last_min = std::numeric_limits<float>::max();
    m_last_max = -std::numeric_limits<float>::max();
    if (m_functions[MAX_POS])
        m_last_max_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_functions[MIN_POS])
        m_last_min_pos = std::numeric_limits<double>::quiet_NaN();

    for (size_t i = m_cur_idx; i < m_intervals.size(); ++i) {
        const GInterval &cur = m_intervals[i];
        if (!cur.do_overlap(interval))
            break;

        float v = m_vals[i];
        if (!std::isnan(v)) {
            m_last_sum += v;
            if (v < m_last_min) {
                m_last_min = v;
                if (m_functions[MIN_POS])
                    m_last_min_pos = cur.start;
            } else if (m_functions[MIN_POS] && v == m_last_min) {
                if (std::isnan(m_last_min_pos) || cur.start < m_last_min_pos)
                    m_last_min_pos = cur.start;
            }
            if (v > m_last_max) {
                m_last_max = v;
                if (m_functions[MAX_POS])
                    m_last_max_pos = cur.start;
            }

            if (m_functions[STDDEV])
                mean_square_sum += v * v;

            if (m_use_quantile)
                m_sp.add(v, s_rnd_func);

            if (m_functions[EXISTS])
                m_last_exists = 1;

            if (m_functions[FIRST] && std::isnan(m_last_first))
                m_last_first = v;

            if (m_functions[FIRST_POS] && std::isnan(m_last_first_pos))
                m_last_first_pos = cur.start;

            if (m_functions[LAST])
                m_last_last = v;

            if (m_functions[LAST_POS])
                m_last_last_pos = cur.start;

            if (m_functions[SAMPLE])
                all_values.push_back(v);
            if (m_functions[SAMPLE_POS])
                all_positions.push_back(cur.start);

            ++num_vs;
        }
    }

    if (m_functions[SIZE])
        m_last_size = num_vs;

    if (m_functions[SAMPLE] && !all_values.empty()) {
        int idx = (int)(s_rnd_func() * all_values.size());
        if (idx >= (int)all_values.size())
            idx = (int)all_values.size() - 1;
        if (idx < 0)
            idx = 0;
        m_last_sample = all_values[idx];
    }

    if (m_functions[SAMPLE_POS] && !all_positions.empty()) {
        int idx = (int)(s_rnd_func() * all_positions.size());
        if (idx >= (int)all_positions.size())
            idx = (int)all_positions.size() - 1;
        if (idx < 0)
            idx = 0;
        m_last_sample_pos = all_positions[idx];
    }

    if (num_vs > 0)
        m_last_avg = m_last_nearest = m_last_sum / num_vs;
    else {
        m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_sum = std::numeric_limits<float>::quiet_NaN();
        if (m_functions[MIN_POS])
            m_last_min_pos = std::numeric_limits<double>::quiet_NaN();
    }

    if (m_functions[STDDEV])
        m_last_stddev = num_vs > 1
            ? std::sqrt(mean_square_sum / (num_vs - 1) - (m_last_avg * (double)m_last_avg) * (num_vs / (num_vs - 1)))
            : std::numeric_limits<float>::quiet_NaN();
}

#endif /* GENOMETRACKSPARSE_H_ */
