#include <cstring>

#include "GenomeTrackArray.h"
#include "TrackIndex.h"

const char *GenomeTrackArray::SLICE_FUNCTION_NAMES[GenomeTrackArray::NUM_S_FUNCS] =
    { "avg", "min", "max", "stddev", "sum", "quantile" };

// On disk each interval-table record is (int64 start, int64 end, int64 vals_pos).
// pymisha/_array_track.py writes vals_pos as int64; on 64-bit Linux this is also
// byte-compatible with R misha's `long`.
const int GenomeTrackArray::RECORD_SIZE = 2 * sizeof(int64_t) + sizeof(int64_t);

GenomeTrackArray::GenomeTrackArray() :
    GenomeTrack1D(ARRAYS),
    m_loaded(false),
    m_intervals_pos(0),
    m_cur_idx(0),
    m_base_offset(0),
    m_slice_function(S_AVG),
    m_slice_percentile(0.5),
    m_last_array_vals_idx((uint64_t)-1)
{
    static_assert(sizeof(ArrayVal) == sizeof(float) + sizeof(uint32_t),
                  "ArrayVal must be packed to 8 bytes for bulk reads");
}

void GenomeTrackArray::init_read(const char *filename, int chromid)
{
    m_loaded = false;
    m_base_offset = 0;
    m_intervals.clear();
    m_vals_pos.clear();
    m_array_vals.clear();
    m_last_array_vals_idx = (uint64_t)-1;
    m_cur_idx = 0;

    // Check for the indexed single-file format first. get_track_index() hits the
    // process-static cache (avoiding a stat() per chromosome on the hot path) and
    // returns nullptr when there is no track.idx.
    const std::string track_dir = GenomeTrack::get_track_dir(filename);
    auto idx = get_track_index(track_dir);

    if (idx) {
        const std::string dat_path = track_dir + "/track.dat";

        if (!m_dat_open || m_dat_path != dat_path || m_dat_mode != "rb") {
            m_bfile.close();
            if (m_bfile.open(dat_path.c_str(), "rb"))
                TGLError<GenomeTrackArray>("Cannot open %s: %s", dat_path.c_str(), strerror(errno));
            m_dat_open = true;
            m_dat_path = dat_path;
            m_dat_mode = "rb";
        }

        auto entry = idx->get_entry(chromid);
        if (!entry || entry->length == 0) {
            // Chromosome absent / empty contig: behaves like an empty track.
            m_chromid = chromid;
            return;
        }

        if (m_bfile.seek(entry->offset, SEEK_SET))
            TGLError<GenomeTrackArray>("Failed to seek to offset %llu in %s",
                                       (unsigned long long)entry->offset, dat_path.c_str());

        int32_t signature = 0;
        if (m_bfile.read(&signature, sizeof(signature)) != sizeof(signature))
            TGLError<GenomeTrackArray>("Failed to read array track header in %s", dat_path.c_str());
        if (signature != GenomeTrack::FORMAT_SIGNATURES[ARRAYS])
            TGLError<GenomeTrackArray>("Invalid array track header in %s", dat_path.c_str());

        m_base_offset = entry->offset;
    } else {
        m_bfile.close();
        m_dat_open = false;
        read_type(filename);  // opens, validates the -8 signature, leaves pos at 4
    }

    m_chromid = chromid;
}

void GenomeTrackArray::read_intervals_map()
{
    if (m_loaded)
        return;

    // After init_read the file is positioned just past the 4-byte signature, at
    // the int64 offset of the interval table.
    if (m_bfile.read(&m_intervals_pos, sizeof(m_intervals_pos)) != sizeof(m_intervals_pos)) {
        if (m_bfile.error())
            TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                       m_bfile.file_name().c_str(), strerror(errno));
        TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());
    }

    if (m_base_offset > 0)
        m_intervals_pos += m_base_offset;

    if (m_bfile.seek(m_intervals_pos, SEEK_SET))
        TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                   m_bfile.file_name().c_str(), strerror(errno));

    uint64_t num_intervals = 0;
    if (m_bfile.read(&num_intervals, sizeof(num_intervals)) != sizeof(num_intervals)) {
        if (m_bfile.error())
            TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                       m_bfile.file_name().c_str(), strerror(errno));
        TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());
    }

    m_intervals.resize(num_intervals);
    m_vals_pos.resize(num_intervals);

    for (int64_t i = 0; i < (int64_t)num_intervals; ++i) {
        GInterval &interval = m_intervals[i];

        if (m_bfile.read(&interval.start, sizeof(int64_t)) != sizeof(int64_t) ||
            m_bfile.read(&interval.end, sizeof(int64_t)) != sizeof(int64_t) ||
            m_bfile.read(&m_vals_pos[i], sizeof(int64_t)) != sizeof(int64_t)) {
            if (m_bfile.error())
                TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                           m_bfile.file_name().c_str(), strerror(errno));
            TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());
        }

        if (m_base_offset > 0)
            m_vals_pos[i] += m_base_offset;

        interval.chromid = m_chromid;

        if (interval.start < 0 || interval.start >= interval.end ||
            (i && interval.start < m_intervals[i - 1].end) ||
            m_vals_pos[i] < 0 || (i && m_vals_pos[i - 1] >= m_vals_pos[i]))
            TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());
    }

    m_cur_idx = 0;
    m_loaded = true;
}

void GenomeTrackArray::read_array_vals(uint64_t idx)
{
    if (m_last_array_vals_idx == idx)
        return;
    m_last_array_vals_idx = idx;

    if (m_bfile.seek(m_vals_pos[idx], SEEK_SET))
        TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                   m_bfile.file_name().c_str(), strerror(errno));

    uint32_t num_vals = 0;
    if (m_bfile.read(&num_vals, sizeof(num_vals)) != sizeof(num_vals))
        TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());

    m_array_vals.resize(num_vals);
    if (num_vals > 0) {
        size_t total_bytes = (size_t)num_vals * sizeof(ArrayVal);
        if (m_bfile.read(m_array_vals.data(), total_bytes) != total_bytes) {
            if (m_bfile.error())
                TGLError<GenomeTrackArray>("Failed to read array track file %s: %s",
                                           m_bfile.file_name().c_str(), strerror(errno));
            TGLError<GenomeTrackArray>("Invalid format of array track file %s", m_bfile.file_name().c_str());
        }
    }
}

// Reduce one bin's value list to a scalar, per the slice function. Ported from
// misha's GenomeTrackArrays::get_sliced_val (all-columns and column-subset).
float GenomeTrackArray::get_sliced_val(uint64_t idx)
{
    read_array_vals(idx);

    if (m_slice.empty()) {
        switch (m_slice_function) {
        case S_AVG: {
            if (m_array_vals.empty())
                return std::numeric_limits<float>::quiet_NaN();
            double sum = 0;
            for (const auto &av : m_array_vals)
                sum += av.val;
            return (float)(sum / m_array_vals.size());
        }
        case S_MIN: {
            float s_min = std::numeric_limits<float>::max();
            for (const auto &av : m_array_vals)
                s_min = std::min(av.val, s_min);
            return s_min;
        }
        case S_MAX: {
            float s_max = -std::numeric_limits<float>::max();
            for (const auto &av : m_array_vals)
                s_max = std::max(av.val, s_max);
            return s_max;
        }
        case S_STDDEV: {
            if (m_array_vals.size() <= 1)
                return std::numeric_limits<float>::quiet_NaN();
            long double mean_square_sum = 0;
            double sum = 0;
            for (const auto &av : m_array_vals) {
                sum += av.val;
                // R computes val*val in float32 (catastrophic rounding at large
                // magnitudes); reproduce it bit-for-bit (do NOT widen to double).
                mean_square_sum += av.val * av.val;
            }
            double N = m_array_vals.size();
            double avg = sum / N;
            return (float)std::sqrt(mean_square_sum / (N - 1) - (avg * avg) * (N / (N - 1)));
        }
        case S_SUM: {
            double sum = 0;
            for (const auto &av : m_array_vals)
                sum += av.val;
            return (float)sum;
        }
        case S_QUANTILE: {
            m_slice_sp.reset();
            for (const auto &av : m_array_vals)
                m_slice_sp.add(av.val, s_rnd_func);
            bool is_estimated;
            return m_slice_sp.get_percentile(m_slice_percentile, is_estimated);
        }
        default:
            TGLError<GenomeTrackArray>("Unrecognized slice function");
        }
    }

    switch (m_slice_function) {
    case S_AVG: {
        double sum = 0;
        double N = 0;
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            if (!std::isnan(v)) { sum += v; ++N; }
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        return N ? (float)(sum / N) : std::numeric_limits<float>::quiet_NaN();
    }
    case S_MIN: {
        float s_min = std::numeric_limits<float>::max();
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            if (!std::isnan(v)) s_min = std::min(v, s_min);
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        return s_min == std::numeric_limits<float>::max() ? std::numeric_limits<float>::quiet_NaN() : s_min;
    }
    case S_MAX: {
        float s_max = -std::numeric_limits<float>::max();
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            if (!std::isnan(v)) s_max = std::max(v, s_max);
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        return s_max == -std::numeric_limits<float>::max() ? std::numeric_limits<float>::quiet_NaN() : s_max;
    }
    case S_STDDEV: {
        double mean_square_sum = 0;
        double sum = 0;
        double N = 0;
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            // R accumulates val*val computed in float32 here (the same one-pass
            // formula as the no-slice path); match it bit-for-bit.
            if (!std::isnan(v)) { ++N; sum += v; mean_square_sum += v * v; }
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        if (N <= 1)
            return std::numeric_limits<float>::quiet_NaN();
        double avg = sum / N;
        return (float)std::sqrt(mean_square_sum / (N - 1) - (avg * avg) * (N / (N - 1)));
    }
    case S_SUM: {
        double sum = 0;
        double N = 0;
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            if (!std::isnan(v)) { sum += v; ++N; }
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        return N ? (float)sum : std::numeric_limits<float>::quiet_NaN();
    }
    case S_QUANTILE: {
        m_slice_sp.reset();
        for (uint64_t islice = 0; islice < m_slice.size(); ++islice) {
            float v = get_array_val(islice);
            if (!std::isnan(v)) m_slice_sp.add(v, s_rnd_func);
            if (m_array_hints[islice] >= m_array_vals.size()) break;
        }
        if (m_slice_sp.stream_size()) {
            bool is_estimated;
            return m_slice_sp.get_percentile(m_slice_percentile, is_estimated);
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    default:
        TGLError<GenomeTrackArray>("Unrecognized slice function");
    }
    return std::numeric_limits<float>::quiet_NaN();
}

// Aggregate the per-bin sliced scalars over every bin overlapping the interval,
// starting from the sequential cursor m_cur_idx (positioned by read_interval).
void GenomeTrackArray::calc_vals(const GInterval &interval)
{
    float num_vs = 0;
    double sum = 0;
    double stddev_mean = 0;
    double stddev_m2 = 0;
    m_last_min = std::numeric_limits<float>::max();
    m_last_max = -std::numeric_limits<float>::max();
    if (has_function(MAX_POS))
        m_last_max_pos = std::numeric_limits<double>::quiet_NaN();
    if (has_function(MIN_POS))
        m_last_min_pos = std::numeric_limits<double>::quiet_NaN();

    for (size_t i = m_cur_idx; i < m_intervals.size(); ++i) {
        const GInterval &cur = m_intervals[i];
        if (!cur.do_overlap(interval))
            break;

        float v = get_sliced_val(i);
        if (std::isnan(v))
            continue;

        sum += v;
        if (v < m_last_min) {
            m_last_min = v;
            if (has_function(MIN_POS))
                m_last_min_pos = cur.start;
        }
        if (v > m_last_max) {
            m_last_max = v;
            if (has_function(MAX_POS))
                m_last_max_pos = cur.start;
        }

        if (has_function(EXISTS))
            m_last_exists = 1;
        if (has_function(FIRST) && std::isnan(m_last_first))
            m_last_first = v;
        if (has_function(FIRST_POS) && std::isnan(m_last_first_pos))
            m_last_first_pos = cur.start;
        if (has_function(LAST))
            m_last_last = v;
        if (has_function(LAST_POS))
            m_last_last_pos = cur.start;

        ++num_vs;
        if (has_function(STDDEV)) {
            const double delta = v - stddev_mean;
            stddev_mean += delta / (double)num_vs;
            const double delta2 = v - stddev_mean;
            stddev_m2 += delta * delta2;
        }
        if (m_use_quantile)
            m_sp.add(v, s_rnd_func);
    }

    if (has_function(SIZE))
        m_last_size = num_vs;

    m_last_sum = (float)sum;
    if (num_vs > 0)
        m_last_avg = m_last_nearest = (float)(sum / num_vs);
    else {
        m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_sum =
            std::numeric_limits<float>::quiet_NaN();
        if (has_function(MIN_POS))
            m_last_min_pos = std::numeric_limits<double>::quiet_NaN();
    }

    if (has_function(STDDEV))
        m_last_stddev = num_vs > 1
            ? (float)std::sqrt(stddev_m2 / (double)(num_vs - 1))
            : std::numeric_limits<float>::quiet_NaN();
}

void GenomeTrackArray::read_interval(const GInterval &interval)
{
    m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_stddev = m_last_sum =
        std::numeric_limits<float>::quiet_NaN();
    if (has_function(MAX_POS))
        m_last_max_pos = std::numeric_limits<double>::quiet_NaN();
    if (has_function(MIN_POS))
        m_last_min_pos = std::numeric_limits<double>::quiet_NaN();
    if (has_function(EXISTS))
        m_last_exists = 0;
    if (has_function(SIZE))
        m_last_size = 0;
    if (has_function(FIRST))
        m_last_first = std::numeric_limits<float>::quiet_NaN();
    if (has_function(FIRST_POS))
        m_last_first_pos = std::numeric_limits<double>::quiet_NaN();
    if (has_function(LAST))
        m_last_last = std::numeric_limits<float>::quiet_NaN();
    if (has_function(LAST_POS))
        m_last_last_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_use_quantile)
        m_sp.reset();

    if (!m_loaded)
        read_intervals_map();

    if (m_intervals.empty())
        return;

    if (m_intervals.front().start >= interval.end) {
        m_last_nearest = get_sliced_val(0);
        return;
    }
    if (m_intervals.back().end <= interval.start) {
        m_last_nearest = get_sliced_val(m_intervals.size() - 1);
        return;
    }

    if (check_first_overlap(m_cur_idx, interval)) {
        calc_vals(interval);
        return;
    }
    if (m_cur_idx + 1 < m_intervals.size() && check_first_overlap(m_cur_idx + 1, interval)) {
        ++m_cur_idx;
        calc_vals(interval);
        return;
    }

    size_t istart = 0;
    size_t iend = m_intervals.size();
    while (iend - istart > 1) {
        size_t imid = istart + (iend - istart) / 2;
        if (check_first_overlap(imid, interval)) {
            m_cur_idx = imid;
            calc_vals(interval);
            return;
        }
        if (m_intervals[imid].start < interval.start)
            istart = imid;
        else
            iend = imid;
    }

    if (iend - istart == 1 && check_first_overlap(istart, interval)) {
        m_cur_idx = istart;
        calc_vals(interval);
        return;
    }

    if (iend - istart == 1) {
        const GInterval &left = m_intervals[istart];
        const GInterval &right = (iend < m_intervals.size()) ? m_intervals[iend] : left;
        double left_dist = interval.dist2interv(left);
        double right_dist = interval.dist2interv(right);
        if (iend >= m_intervals.size() || left_dist <= right_dist)
            m_last_nearest = get_sliced_val(istart);
        else
            m_last_nearest = get_sliced_val(iend);
    }
}

const std::vector<GInterval> &GenomeTrackArray::get_intervals()
{
    read_intervals_map();
    return m_intervals;
}
