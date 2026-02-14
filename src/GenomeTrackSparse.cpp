#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

#include "GenomeTrackSparse.h"
#include "TrackIndex.h"

const int GenomeTrackSparse::RECORD_SIZE = 2 * sizeof(int64_t) + sizeof(float);

GenomeTrackSparse::GenomeTrackSparse() :
    GenomeTrack1D(SPARSE),
    m_loaded(false),
    m_num_records(0),
    m_cur_idx(0),
    m_last_min_pos(std::numeric_limits<double>::quiet_NaN())
{}

void GenomeTrackSparse::read_header_at_current_pos_(BufferedFile &bf)
{
    int32_t format_signature = 0;
    if (bf.read(&format_signature, sizeof(format_signature)) != sizeof(format_signature))
        TGLError<GenomeTrackSparse>("Corrupt sparse header in %s", bf.file_name().c_str());
    if (format_signature >= 0)
        TGLError<GenomeTrackSparse>("Invalid sparse header signature in %s", bf.file_name().c_str());
}

void GenomeTrackSparse::init_read(const char *filename, int chromid)
{
    m_loaded = false;
    uint64_t header_start = 0;
    uint64_t total_bytes = 0;

    const std::string track_dir = GenomeTrack::get_track_dir(filename);
    const std::string idx_path = track_dir + "/track.idx";

    struct stat idx_st;
    if (stat(idx_path.c_str(), &idx_st) == 0) {
        const std::string dat_path = track_dir + "/track.dat";

        if (!m_dat_open || m_dat_path != dat_path || m_dat_mode != "rb") {
            m_bfile.close();
            if (m_bfile.open(dat_path.c_str(), "rb"))
                TGLError<GenomeTrackSparse>("Cannot open %s: %s", dat_path.c_str(), strerror(errno));
            m_dat_open = true;
            m_dat_path = dat_path;
            m_dat_mode = "rb";
        }

        auto idx = get_track_index(track_dir);
        if (!idx)
            TGLError<GenomeTrackSparse>("Failed to load track index for %s", track_dir.c_str());

        auto entry = idx->get_entry(chromid);
        if (!entry || entry->length == 0) {
            m_num_records = 0;
            m_chromid = chromid;
            return;
        }

        if (m_bfile.seek(entry->offset, SEEK_SET))
            TGLError<GenomeTrackSparse>("Failed to seek to offset %llu in %s",
                                        (unsigned long long)entry->offset, dat_path.c_str());

        header_start = entry->offset;
        read_header_at_current_pos_(m_bfile);
        total_bytes = entry->length;
    } else {
        m_bfile.close();
        m_dat_open = false;
        read_type(filename);

        header_start = 0;
        total_bytes = m_bfile.file_size();
    }

    const uint64_t header_size = m_bfile.tell() - header_start;
    const double n = (total_bytes - header_size) / (double)kSparseRecBytes;

    if (n != (int64_t)n)
        TGLError<GenomeTrackSparse>(
            "Invalid sparse track file %s: file size (%llu bytes) minus header (%llu bytes) "
            "is not divisible by record size (%zu bytes). Computed %.2f records.",
            filename, (unsigned long long)total_bytes, (unsigned long long)header_size, kSparseRecBytes, n);

    m_num_records = (int64_t)n;
    m_chromid = chromid;
}

void GenomeTrackSparse::init_write(const char *filename, int chromid)
{
    m_bfile.close();
    m_loaded = false;
    write_type(filename);
    m_chromid = chromid;
}

void GenomeTrackSparse::read_file_into_mem()
{
    if (m_loaded)
        return;

    m_intervals.resize(m_num_records);
    m_vals.resize(m_num_records);

    if (m_num_records == 0) {
        m_cur_idx = 0;
        m_loaded = true;
        return;
    }

    // Read records directly into target arrays, one at a time, avoiding bulk buffer allocation
    char rec[kSparseRecBytes];
    for (int64_t i = 0; i < m_num_records; ++i) {
        uint64_t bytes_read = m_bfile.read(rec, kSparseRecBytes);
        if (bytes_read != kSparseRecBytes) {
            if (m_bfile.error())
                TGLError<GenomeTrackSparse>("Failed to read sparse track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
            TGLError<GenomeTrackSparse>("Truncated sparse track file %s at record %lld",
                                        m_bfile.file_name().c_str(), (long long)i);
        }

        GInterval &interval = m_intervals[i];
        memcpy(&interval.start, rec, sizeof(int64_t));
        memcpy(&interval.end, rec + sizeof(int64_t), sizeof(int64_t));
        memcpy(&m_vals[i], rec + 2 * sizeof(int64_t), sizeof(float));

        if (isinf(m_vals[i]))
            m_vals[i] = std::numeric_limits<float>::quiet_NaN();

        interval.chromid = m_chromid;

        if (interval.start < 0) {
            TGLError<GenomeTrackSparse>("Invalid sparse track file %s: interval %lld has negative start (%lld)",
                                        m_bfile.file_name().c_str(), (long long)i, (long long)interval.start);
        }
        if (interval.start >= interval.end) {
            TGLError<GenomeTrackSparse>("Invalid sparse track file %s: interval %lld has start (%lld) >= end (%lld)",
                                        m_bfile.file_name().c_str(), (long long)i, (long long)interval.start, (long long)interval.end);
        }
        if (i && interval.start < m_intervals[i - 1].end) {
            TGLError<GenomeTrackSparse>("Invalid sparse track file %s: interval %lld [%lld, %lld) overlaps with previous interval [%lld, %lld)",
                                        m_bfile.file_name().c_str(), (long long)i,
                                        (long long)interval.start, (long long)interval.end,
                                        (long long)m_intervals[i - 1].start, (long long)m_intervals[i - 1].end);
        }
    }

    m_cur_idx = 0;
    m_loaded = true;
}

void GenomeTrackSparse::read_interval(const GInterval &interval)
{
    m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_stddev = m_last_sum = std::numeric_limits<float>::quiet_NaN();
    if (m_functions[MAX_POS])
        m_last_max_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_functions[MIN_POS])
        m_last_min_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_functions[EXISTS])
        m_last_exists = 0;
    if (m_functions[SIZE])
        m_last_size = 0;
    if (m_functions[SAMPLE])
        m_last_sample = std::numeric_limits<float>::quiet_NaN();
    if (m_functions[SAMPLE_POS])
        m_last_sample_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_functions[FIRST])
        m_last_first = std::numeric_limits<float>::quiet_NaN();
    if (m_functions[FIRST_POS])
        m_last_first_pos = std::numeric_limits<double>::quiet_NaN();
    if (m_functions[LAST])
        m_last_last = std::numeric_limits<float>::quiet_NaN();
    if (m_functions[LAST_POS])
        m_last_last_pos = std::numeric_limits<double>::quiet_NaN();

    if (m_use_quantile)
        m_sp.reset();

    read_file_into_mem();

    if (m_intervals.empty())
        return;

    if (m_intervals.front().start >= interval.end) {
        m_last_nearest = m_vals.front();
        return;
    }

    if (m_intervals.back().end <= interval.start) {
        m_last_nearest = m_vals.back();
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
            m_last_nearest = m_vals[istart];
        else
            m_last_nearest = m_vals[iend];
    }
}

double GenomeTrackSparse::last_max_pos() const
{
    return m_last_max_pos;
}

double GenomeTrackSparse::last_min_pos() const
{
    return m_last_min_pos;
}

void GenomeTrackSparse::write_next_interval(const GInterval &interval, float val)
{
    uint64_t size = 0;
    size += m_bfile.write(&interval.start, sizeof(interval.start));
    size += m_bfile.write(&interval.end, sizeof(interval.end));
    size += m_bfile.write(&val, sizeof(val));

    if ((int)size != RECORD_SIZE) {
        if (m_bfile.error())
            TGLError<GenomeTrackSparse>("Failed to write a sparse track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
        TGLError<GenomeTrackSparse>("Failed to write a sparse track file %s", m_bfile.file_name().c_str());
    }
}

const std::vector<GInterval> &GenomeTrackSparse::get_intervals()
{
    read_file_into_mem();
    return m_intervals;
}

const std::vector<float> &GenomeTrackSparse::get_vals()
{
    read_file_into_mem();
    return m_vals;
}
