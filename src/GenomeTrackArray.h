/*
 * GenomeTrackArray.h
 *
 * Array (multi-column 1D) track reader for pymisha, ported from misha's
 * GenomeTrackArrays. Reads the same on-disk format the pure-Python reader in
 * pymisha/_array_track.py describes (format signature -8):
 *
 *   int32   signature (-8)
 *   int64   intervals_pos  (offset to the interval table)
 *   <value blocks at varying offsets, one per bin>:
 *       uint32 num_vals
 *       num_vals * { float32 val; uint32 idx }
 *   at intervals_pos:
 *       uint64 num_intervals
 *       num_intervals * { int64 start; int64 end; int64 vals_pos }
 *
 * Indexed single-file tracks (track.idx + track.dat) store each chromosome's
 * verbatim block in track.dat; offsets inside it are block-relative and are
 * adjusted by m_base_offset, exactly like GenomeTrackSparse.
 *
 * Each bin carries a sparse set of {value, column-index} records. A per-bin
 * scalar is produced by the slice reduction (avg/min/max/stddev/sum/quantile)
 * over either all columns or a selected column subset; read_interval then
 * aggregates those per-bin scalars over every bin overlapping the query
 * interval (Welford), matching the sparse track reducers.
 */

#ifndef GENOMETRACKARRAY_H_
#define GENOMETRACKARRAY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "GenomeTrack1D.h"
#include "GInterval.h"

// !!!!!!!!! IN CASE OF ERROR THIS CLASS THROWS TGLException  !!!!!!!!!!!!!!!!

class GenomeTrackArray : public GenomeTrack1D {
public:
    enum SliceFunctions { S_AVG, S_MIN, S_MAX, S_STDDEV, S_SUM, S_QUANTILE, NUM_S_FUNCS };

    static const char *SLICE_FUNCTION_NAMES[NUM_S_FUNCS];

#pragma pack(push, 1)
    struct ArrayVal {
        float    val;
        uint32_t idx;

        ArrayVal() {}
        ArrayVal(float _val, uint32_t _idx) : val(_val), idx(_idx) {}
    };
#pragma pack(pop)

    struct LessIdx {
        bool operator()(const ArrayVal &obj, uint32_t idx) const { return obj.idx < idx; }
    };

    typedef std::vector<ArrayVal> ArrayVals;

    GenomeTrackArray();

    void read_interval(const GInterval &interval) override;
    double last_max_pos() const override { return m_last_max_pos; }
    double last_min_pos() const override { return m_last_min_pos; }

    void init_read(const char *filename, int chromid);

    // Column selection + reduction for the per-bin scalar. An empty slice means
    // "use every column"; a non-empty slice selects (0-based) column indices.
    void set_slice(const std::vector<uint32_t> &slice);
    void set_slice_function(SliceFunctions func, const std::vector<uint32_t> &slice);
    void set_slice_quantile(double percentile, uint64_t rnd_sampling_buf_size,
                            uint64_t lowest_vals_buf_size, uint64_t highest_vals_buf_size,
                            const std::vector<uint32_t> &slice);

    // Bin coordinates, for use as a scanner iterator (one interval per bin).
    const std::vector<GInterval> &get_intervals();

protected:
    static const int RECORD_SIZE;  // bytes per interval-table record

    std::vector<GInterval> m_intervals;
    std::vector<int64_t>   m_vals_pos;
    bool                   m_loaded;
    int64_t                m_intervals_pos;
    size_t                 m_cur_idx;       // sequential overlap cursor
    int64_t                m_base_offset;   // >0 for indexed track.dat slices

    // State for the indexed "smart handle" (mirrors GenomeTrackSparse).
    std::string m_dat_path;
    std::string m_dat_mode;
    bool        m_dat_open{false};

    // Slice configuration.
    SliceFunctions           m_slice_function;
    double                   m_slice_percentile;
    std::vector<uint32_t>    m_slice;
    std::vector<uint32_t>    m_array_hints;
    StreamPercentiler<float> m_slice_sp;

    // Lazily-read value block for the current bin.
    ArrayVals m_array_vals;
    uint64_t  m_last_array_vals_idx;

    void  read_intervals_map();
    void  read_array_vals(uint64_t idx);
    float get_array_val(uint64_t islice);
    float get_sliced_val(uint64_t idx);
    void  calc_vals(const GInterval &interval);
    bool  check_first_overlap(size_t idx, const GInterval &interval) const;
};

//------------------------------------ IMPLEMENTATION --------------------------------

inline void GenomeTrackArray::set_slice(const std::vector<uint32_t> &slice)
{
    m_slice = slice;
    m_array_hints.assign(m_slice.size(), 0);
}

inline void GenomeTrackArray::set_slice_function(SliceFunctions func, const std::vector<uint32_t> &slice)
{
    m_slice_function = func;
    set_slice(slice);
}

inline void GenomeTrackArray::set_slice_quantile(double percentile, uint64_t rnd_sampling_buf_size,
                                                 uint64_t lowest_vals_buf_size, uint64_t highest_vals_buf_size,
                                                 const std::vector<uint32_t> &slice)
{
    m_slice_function = S_QUANTILE;
    m_slice_percentile = percentile;
    m_slice_sp.init(rnd_sampling_buf_size, lowest_vals_buf_size, highest_vals_buf_size);
    set_slice(slice);
}

inline bool GenomeTrackArray::check_first_overlap(size_t idx, const GInterval &interval) const
{
    if (idx >= m_intervals.size())
        return false;
    const GInterval &cur = m_intervals[idx];
    return cur.do_overlap(interval) && (idx == 0 || !m_intervals[idx - 1].do_overlap(interval));
}

// Locate one slice column inside the current bin's sparse value list, using
// R's hint-accelerated lookup (sequential scans across adjacent slice columns
// are the common case; falls back to binary search by column index).
inline float GenomeTrackArray::get_array_val(uint64_t islice)
{
    uint32_t &hint = m_array_hints[islice];
    uint32_t slice = m_slice[islice];

    if (hint < m_array_vals.size() && m_array_vals[hint].idx == slice)
        return m_array_vals[hint].val;

    uint32_t prev_hint;
    if (islice) {
        prev_hint = m_array_hints[islice - 1];
        hint = prev_hint + 1;
        if (hint < m_array_vals.size() && m_array_vals[hint].idx == slice)
            return m_array_vals[hint].val;
    } else
        prev_hint = 0;

    ArrayVals::const_iterator it = std::lower_bound(
        m_array_vals.begin() + prev_hint, m_array_vals.end(), slice, LessIdx());
    hint = (uint32_t)(it - m_array_vals.begin());
    return it < m_array_vals.end() && it->idx == slice
        ? it->val : std::numeric_limits<float>::quiet_NaN();
}

#endif /* GENOMETRACKARRAY_H_ */
