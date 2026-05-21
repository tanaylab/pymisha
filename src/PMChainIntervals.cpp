// pm_chain_intervals_resolve: C++ port of liftover._handle_src_overlaps +
// _handle_tgt_overlaps (and the helpers _discard_overlapping_intervals,
// _handle_tgt_overlaps_auto, _handle_tgt_overlaps_agg, _sweep_line_winners).
//
// Single entry point covers both src + tgt overlap-policy passes per
// gintervals_load_chain's needs. Out-of-scope for this port: map_interval,
// buildSrcAux (G1.P3.B.2), cluster strategies best_cluster_* (G1.P3.B.2 +
// stays Python until then).

#include "pymisha.h"
#include "PMChainIntervals.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <set>            // used by handle_tgt_overlaps in Tasks 4-6
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ChainRow {
    int     chromid;       // dense id into tgt chrom table
    int     chromid_src;   // dense id into src chrom table
    int64_t start;
    int64_t end;
    int64_t strand;
    int64_t start_src;
    int64_t end_src;
    int64_t strand_src;
    int64_t chain_id;
    double  score;
};

// Pull a key from df_dict and coerce to a contiguous 1-D numpy array of the
// requested dtype. Returns nullptr on failure (with Python error set).
PyObject *get_array(PyObject *df_dict, const char *key, int dtype)
{
    PyObject *item = PyDict_GetItemString(df_dict, key);
    if (!item) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict missing required key '%s'", key);
        return nullptr;
    }
    PyObject *arr = PyArray_FROM_OTF(item, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return nullptr;
    if (PyArray_NDIM((PyArrayObject *)arr) != 1) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict['%s'] must be 1-D", key);
        Py_DECREF(arr);
        return nullptr;
    }
    return arr;
}

bool valid_src_policy(const char *p)
{
    return !strcmp(p, "error") || !strcmp(p, "keep") || !strcmp(p, "discard");
}

bool valid_tgt_policy(const char *p)
{
    return !strcmp(p, "error") || !strcmp(p, "keep") || !strcmp(p, "discard") ||
           !strcmp(p, "auto") || !strcmp(p, "auto_first") ||
           !strcmp(p, "auto_longer") || !strcmp(p, "auto_score") ||
           !strcmp(p, "agg");
}

class ChromInterner {
public:
    int intern(const std::string &s) {
        auto it = m_map.find(s);
        if (it != m_map.end()) return it->second;
        int id = (int)m_names.size();
        m_map.emplace(s, id);
        m_names.push_back(s);
        return id;
    }
    int find_existing(const std::string &s) const {
        auto it = m_map.find(s);
        if (it == m_map.end()) return -1;
        return it->second;
    }
    const std::string &name_of(int id) const { return m_names[(size_t)id]; }
private:
    std::unordered_map<std::string, int> m_map;
    std::vector<std::string> m_names;
};

PMPY rows_to_dict(const std::vector<ChainRow> &rows,
                  const ChromInterner &tgt_chrom,
                  const ChromInterner &src_chrom)
{
    npy_intp n_out = (npy_intp)rows.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n_out, NPY_OBJECT), true);
    PMPY py_chromsrc(PyArray_SimpleNew(1, &n_out, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_strand(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_startsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_endsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_strandsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_chain_id(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_score(PyArray_SimpleNew(1, &n_out, NPY_DOUBLE), true);
    if (!py_chrom || !py_chromsrc || !py_start || !py_end || !py_strand
        || !py_startsrc || !py_endsrc || !py_strandsrc
        || !py_chain_id || !py_score) {
        return PMPY();
    }

    PyObject **chrom_out    = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    PyObject **chromsrc_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chromsrc);
    int64_t *start_out      = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out        = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    int64_t *strand_out     = (int64_t *)PyArray_DATA((PyArrayObject *)*py_strand);
    int64_t *startsrc_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_startsrc);
    int64_t *endsrc_out     = (int64_t *)PyArray_DATA((PyArrayObject *)*py_endsrc);
    int64_t *strandsrc_out  = (int64_t *)PyArray_DATA((PyArrayObject *)*py_strandsrc);
    int64_t *chain_id_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_chain_id);
    double *score_out       = (double *)PyArray_DATA((PyArrayObject *)*py_score);

    for (npy_intp i = 0; i < n_out; ++i) {
        chrom_out[i] = nullptr;
        chromsrc_out[i] = nullptr;
    }

    for (npy_intp i = 0; i < n_out; ++i) {
        const std::string &cs = tgt_chrom.name_of(rows[i].chromid);
        PyObject *s1 = PyUnicode_FromStringAndSize(cs.data(), cs.size());
        if (!s1) {
            for (npy_intp j = i; j < n_out; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
                Py_INCREF(Py_None); chromsrc_out[j] = Py_None;
            }
            return PMPY();
        }
        chrom_out[i] = s1;
        const std::string &cs2 = src_chrom.name_of(rows[i].chromid_src);
        PyObject *s2 = PyUnicode_FromStringAndSize(cs2.data(), cs2.size());
        if (!s2) {
            for (npy_intp j = i + 1; j < n_out; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
            }
            for (npy_intp j = i; j < n_out; ++j) {
                Py_INCREF(Py_None); chromsrc_out[j] = Py_None;
            }
            return PMPY();
        }
        chromsrc_out[i] = s2;

        start_out[i]     = rows[i].start;
        end_out[i]       = rows[i].end;
        strand_out[i]    = rows[i].strand;
        startsrc_out[i]  = rows[i].start_src;
        endsrc_out[i]    = rows[i].end_src;
        strandsrc_out[i] = rows[i].strand_src;
        chain_id_out[i]  = rows[i].chain_id;
        score_out[i]     = rows[i].score;
    }

    PMPY result(PyDict_New(), true);
    if (!result) return PMPY();
    PyDict_SetItemString(result, "chrom",     py_chrom);
    PyDict_SetItemString(result, "start",     py_start);
    PyDict_SetItemString(result, "end",       py_end);
    PyDict_SetItemString(result, "strand",    py_strand);
    PyDict_SetItemString(result, "chromsrc",  py_chromsrc);
    PyDict_SetItemString(result, "startsrc",  py_startsrc);
    PyDict_SetItemString(result, "endsrc",    py_endsrc);
    PyDict_SetItemString(result, "strandsrc", py_strandsrc);
    PyDict_SetItemString(result, "chain_id",  py_chain_id);
    PyDict_SetItemString(result, "score",     py_score);

    return result;
}

bool dict_to_rows(PyObject *df_dict,
                  std::vector<ChainRow> &out_rows,
                  ChromInterner &tgt_chrom,
                  ChromInterner &src_chrom)
{
    PMPY arr_chrom(get_array(df_dict, "chrom", NPY_OBJECT), true);
    if (!arr_chrom) return false;
    PMPY arr_chromsrc(get_array(df_dict, "chromsrc", NPY_OBJECT), true);
    if (!arr_chromsrc) return false;
    PMPY arr_start(get_array(df_dict, "start", NPY_INT64), true);
    if (!arr_start) return false;
    PMPY arr_end(get_array(df_dict, "end", NPY_INT64), true);
    if (!arr_end) return false;
    PMPY arr_strand(get_array(df_dict, "strand", NPY_INT64), true);
    if (!arr_strand) return false;
    PMPY arr_startsrc(get_array(df_dict, "startsrc", NPY_INT64), true);
    if (!arr_startsrc) return false;
    PMPY arr_endsrc(get_array(df_dict, "endsrc", NPY_INT64), true);
    if (!arr_endsrc) return false;
    PMPY arr_strandsrc(get_array(df_dict, "strandsrc", NPY_INT64), true);
    if (!arr_strandsrc) return false;
    PMPY arr_chain_id(get_array(df_dict, "chain_id", NPY_INT64), true);
    if (!arr_chain_id) return false;
    PMPY arr_score(get_array(df_dict, "score", NPY_DOUBLE), true);
    if (!arr_score) return false;

    const npy_intp n_rows = PyArray_DIM((PyArrayObject *)*arr_chrom, 0);
    PyArrayObject *checks[] = {
        (PyArrayObject *)*arr_chromsrc, (PyArrayObject *)*arr_start,
        (PyArrayObject *)*arr_end, (PyArrayObject *)*arr_strand,
        (PyArrayObject *)*arr_startsrc, (PyArrayObject *)*arr_endsrc,
        (PyArrayObject *)*arr_strandsrc, (PyArrayObject *)*arr_chain_id,
        (PyArrayObject *)*arr_score,
    };
    for (PyArrayObject *a : checks) {
        if (PyArray_DIM(a, 0) != n_rows) {
            PyErr_SetString(PyExc_ValueError, "chain_df_dict columns have mismatched lengths");
            return false;
        }
    }

    PyObject **chrom_in    = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chrom);
    PyObject **chromsrc_in = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chromsrc);
    const int64_t *start_in     = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_start);
    const int64_t *end_in       = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_end);
    const int64_t *strand_in    = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_strand);
    const int64_t *startsrc_in  = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_startsrc);
    const int64_t *endsrc_in    = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_endsrc);
    const int64_t *strandsrc_in = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_strandsrc);
    const int64_t *chain_id_in  = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_chain_id);
    const double *score_in      = (const double *)PyArray_DATA((PyArrayObject *)*arr_score);

    out_rows.clear();
    out_rows.reserve((size_t)n_rows);
    for (npy_intp i = 0; i < n_rows; ++i) {
        if (!chrom_in[i] || !PyUnicode_Check(chrom_in[i])) {
            PyErr_Format(PyExc_TypeError, "chain_df_dict['chrom'][%lld] is not a string", (long long)i);
            return false;
        }
        if (!chromsrc_in[i] || !PyUnicode_Check(chromsrc_in[i])) {
            PyErr_Format(PyExc_TypeError, "chain_df_dict['chromsrc'][%lld] is not a string", (long long)i);
            return false;
        }
        Py_ssize_t clen = 0;
        const char *cstr = PyUnicode_AsUTF8AndSize(chrom_in[i], &clen);
        if (!cstr) return false;
        std::string ckey(cstr, (size_t)clen);
        Py_ssize_t cslen = 0;
        const char *csstr = PyUnicode_AsUTF8AndSize(chromsrc_in[i], &cslen);
        if (!csstr) return false;
        std::string cskey(csstr, (size_t)cslen);

        ChainRow row;
        row.chromid     = tgt_chrom.intern(ckey);
        row.chromid_src = src_chrom.intern(cskey);
        row.start       = start_in[i];
        row.end         = end_in[i];
        row.strand      = strand_in[i];
        row.start_src   = startsrc_in[i];
        row.end_src     = endsrc_in[i];
        row.strand_src  = strandsrc_in[i];
        row.chain_id    = chain_id_in[i];
        row.score       = score_in[i];
        out_rows.push_back(row);
    }
    return true;
}

// Sort rows by (chromid_src, start_src, end_src). Stable for deterministic output.
void sort_by_src(std::vector<ChainRow> &rows)
{
    std::stable_sort(rows.begin(), rows.end(),
        [](const ChainRow &a, const ChainRow &b) {
            if (a.chromid_src != b.chromid_src) return a.chromid_src < b.chromid_src;
            if (a.start_src   != b.start_src)   return a.start_src   < b.start_src;
            return a.end_src < b.end_src;
        });
}

// Sort rows by (chromid, start, end). Stable.
void sort_by_tgt(std::vector<ChainRow> &rows)
{
    std::stable_sort(rows.begin(), rows.end(),
        [](const ChainRow &a, const ChainRow &b) {
            if (a.chromid != b.chromid) return a.chromid < b.chromid;
            if (a.start   != b.start)   return a.start   < b.start;
            return a.end < b.end;
        });
}

void handle_src_overlaps(std::vector<ChainRow> &rows, const std::string &policy,
                         const ChromInterner &/*tgt_chrom*/, const ChromInterner &src_chrom)
{
    if (rows.empty() || policy == "keep") return;

    sort_by_src(rows);

    if (policy == "error") {
        for (size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].chromid_src == rows[i-1].chromid_src &&
                rows[i].start_src < rows[i-1].end_src) {
                PyErr_Format(PyExc_ValueError,
                    "Source overlap detected on %s: [%lld, %lld) overlaps [%lld, %lld)",
                    src_chrom.name_of(rows[i].chromid_src).c_str(),
                    (long long)rows[i-1].start_src, (long long)rows[i-1].end_src,
                    (long long)rows[i].start_src, (long long)rows[i].end_src);
                TGLError("propagating PyErr from handle_src_overlaps");
            }
        }
        return;
    }

    // policy == "discard": R uses a PAIR-ONLY scan after sort_by_src (rdbinterval.cpp:820-841).
    // For each consecutive pair on the same chromsrc, if they overlap mark BOTH.
    // This is strictly weaker than whole-cluster discard: a row nested inside
    // a larger row with a GAP between it and the previous row stays kept.
    // Example: src rows [0,200), [10,50), [60,80) -> R keeps [60,80) because
    // pair (row2, row3) doesn't overlap (50 < 60). **R parity per user direction
    // 2026-05-20.**
    if (policy == "discard") {
        const size_t n = rows.size();
        std::vector<char> discard(n, 0);
        for (size_t i = 1; i < n; ++i) {
            if (rows[i].chromid_src == rows[i-1].chromid_src &&
                rows[i-1].end_src > rows[i].start_src) {
                discard[i-1] = 1;
                discard[i]   = 1;
            }
        }

        // Compact in place: keep only rows where discard[k] == 0.
        size_t write = 0;
        for (size_t k = 0; k < n; ++k) {
            if (!discard[k]) {
                if (write != k) rows[write] = rows[k];
                ++write;
            }
        }
        rows.resize(write);
        return;
    }

    // Unreachable - validated before entry.
    PyErr_Format(PyExc_ValueError, "Unknown src_overlap_policy: %s", policy.c_str());
    TGLError("propagating PyErr from handle_src_overlaps");
}

// Sweep-line event used by tgt-overlap policies. Sort order: position first,
// then "close" events before "open" events at the same position (so an interval
// closing at p=10 vacates the active set BEFORE another opens at p=10), then
// by index for stable behavior.
struct TgtEvent {
    int64_t pos;
    bool    is_start;
    size_t  idx;
    bool operator<(const TgtEvent &other) const {
        if (pos != other.pos) return pos < other.pos;
        if (is_start != other.is_start) return is_start < other.is_start;
        return idx < other.idx;
    }
};

// Given the original ChainRow at index w_idx and a sub-segment [seg_start,
// seg_end), produce a sliced ChainRow with correctly-projected src coords.
// Matches Python's _handle_tgt_overlaps_auto src-coord mapping (lines 644-672)
// and R's append_slice (rdbinterval.cpp:878-905).
//
// Strand semantics: orig.strand == 0 means tgt-strand "+" (src advances
// forward with tgt). orig.strand == 1 means tgt-strand "-" (src retreats as
// tgt advances). NOTE: this is the TGT strand, not the SRC strand.
ChainRow slice_row(const ChainRow &orig, int64_t seg_start, int64_t seg_end)
{
    ChainRow s = orig;
    s.start = seg_start;
    s.end   = seg_end;
    const int64_t orig_tgt_len = orig.end - orig.start;
    if (orig_tgt_len <= 0) {
        // Zero-length target keeps src as-is. Mirrors mask_z in Python.
        s.start_src = orig.start_src;
        s.end_src   = orig.end_src;
    } else if (orig.strand == 0) {
        // Positive strand: src advances forward with tgt.
        const int64_t delta_start = seg_start - orig.start;
        const int64_t delta_end   = seg_end   - orig.start;
        s.start_src = orig.start_src + delta_start;
        s.end_src   = orig.start_src + delta_end;
    } else {
        // Negative strand: src retreats as tgt advances.
        const int64_t delta_end   = seg_end   - orig.start;
        const int64_t delta_start = seg_start - orig.start;
        s.start_src = orig.end_src - delta_end;
        s.end_src   = orig.end_src - delta_start;
    }
    return s;
}

// Try to merge `slice` into the back of `out`. Returns true on merge.
// Mirrors R's append_slice merge predicate (rdbinterval.cpp:889-902): same
// (chromid, chain_id, strand, chromid_src, strand_src), tgt-adjacent
// (prev.end == slice.start), AND src-adjacent (prev.end_src == slice.start_src).
// The src-adjacency requirement is what differentiates this from Python's
// merge step (liftover.py:686-711) which only checks tgt-adjacency + same
// chain_id. For positive-strand splits R and Python agree; for negative-strand
// splits the slices have reversed src coords (prev.end_src != slice.start_src)
// so R refuses to merge - leaving N rows. Python's loose merge collapses them
// back to 1 row via min/max. **R parity wins per user direction 2026-05-20.**
bool try_merge_back(std::vector<ChainRow> &out, const ChainRow &slice)
{
    if (out.empty()) return false;
    ChainRow &prev = out.back();
    if (prev.chromid     != slice.chromid)     return false;
    if (prev.chain_id    != slice.chain_id)    return false;
    if (prev.strand      != slice.strand)      return false;
    if (prev.chromid_src != slice.chromid_src) return false;
    if (prev.strand_src  != slice.strand_src)  return false;
    if (prev.end         != slice.start)       return false;
    if (prev.end_src     != slice.start_src)   return false;
    prev.end     = slice.end;
    prev.end_src = slice.end_src;
    return true;
}

void append_slice(std::vector<ChainRow> &out, const ChainRow &orig,
                  int64_t seg_start, int64_t seg_end, bool allow_merge)
{
    if (seg_end <= seg_start) return;
    ChainRow s = slice_row(orig, seg_start, seg_end);
    if (allow_merge && try_merge_back(out, s)) return;
    out.push_back(s);
}

size_t pick_by_score(const std::vector<ChainRow> &rows, const std::set<size_t> &active)
{
    size_t best = *active.begin();
    for (size_t idx : active) {
        const ChainRow &c = rows[idx];
        const ChainRow &b = rows[best];
        if (c.score > b.score) { best = idx; continue; }
        if (c.score < b.score) continue;
        const int64_t cs = c.end - c.start, bs = b.end - b.start;
        if (cs > bs) { best = idx; continue; }
        if (cs < bs) continue;
        if (c.chain_id < b.chain_id) best = idx;
    }
    return best;
}

size_t pick_by_length(const std::vector<ChainRow> &rows, const std::set<size_t> &active)
{
    size_t best = *active.begin();
    for (size_t idx : active) {
        const ChainRow &c = rows[idx];
        const ChainRow &b = rows[best];
        const int64_t cs = c.end - c.start, bs = b.end - b.start;
        if (cs > bs) { best = idx; continue; }
        if (cs < bs) continue;
        if (c.score > b.score) { best = idx; continue; }
        if (c.score < b.score) continue;
        if (c.chain_id < b.chain_id) best = idx;
    }
    return best;
}

size_t pick_first(const std::vector<ChainRow> &rows, const std::set<size_t> &active)
{
    size_t best = *active.begin();
    for (size_t idx : active) {
        if (rows[idx].chain_id < rows[best].chain_id) best = idx;
    }
    return best;
}

void handle_tgt_overlaps(std::vector<ChainRow> &rows, const std::string &policy,
                         const ChromInterner &tgt_chrom, const ChromInterner &/*src_chrom*/)
{
    if (rows.empty() || policy == "keep") return;

    sort_by_tgt(rows);

    if (policy == "error") {
        for (size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].chromid == rows[i-1].chromid &&
                rows[i].start < rows[i-1].end) {
                PyErr_Format(PyExc_ValueError,
                    "Target overlap detected on %s: [%lld, %lld) overlaps [%lld, %lld)",
                    tgt_chrom.name_of(rows[i].chromid).c_str(),
                    (long long)rows[i-1].start, (long long)rows[i-1].end,
                    (long long)rows[i].start, (long long)rows[i].end);
                TGLError("propagating PyErr from handle_tgt_overlaps");
            }
        }
        return;
    }

    if (policy == "discard") {
        const size_t n = rows.size();
        std::vector<char> discard(n, 0);
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;
            while (j < n && rows[j].chromid == rows[i].chromid) ++j;

            std::vector<TgtEvent> events;
            events.reserve((j - i) * 2);
            for (size_t k = i; k < j; ++k) {
                // Skip zero-length intervals; they contribute nothing to the
                // breakpoint set + must not participate in the active set (see
                // Python _handle_tgt_overlaps_auto: zero-length rows are
                // implicitly dropped by the np.unique/covers construction).
                if (rows[k].start >= rows[k].end) continue;
                events.push_back({ rows[k].start, true,  k });
                events.push_back({ rows[k].end,   false, k });
            }
            std::sort(events.begin(), events.end());

            std::set<size_t> active;
            size_t e = 0;
            while (e < events.size()) {
                int64_t pos = events[e].pos;
                // First close all "close" events at this pos, then open all "open" events.
                while (e < events.size() && events[e].pos == pos && !events[e].is_start) {
                    active.erase(events[e].idx);
                    ++e;
                }
                while (e < events.size() && events[e].pos == pos && events[e].is_start) {
                    active.insert(events[e].idx);
                    ++e;
                }
                if (e >= events.size()) break;
                int64_t next_pos = events[e].pos;
                if (next_pos <= pos || active.size() <= 1) continue;
                // Mark every active interval at a multi-active segment as discarded.
                for (size_t active_idx : active) discard[active_idx] = 1;
            }
            i = j;
        }

        size_t write = 0;
        for (size_t k = 0; k < rows.size(); ++k) {
            if (!discard[k]) {
                if (write != k) rows[write] = rows[k];
                ++write;
            }
        }
        rows.resize(write);
        return;
    }

    const bool is_auto = (policy == "auto_first" || policy == "auto_longer" ||
                          policy == "auto_score");
    const bool is_agg  = (policy == "agg");
    if (!is_auto && !is_agg) {
        PyErr_Format(PyExc_ValueError, "Unknown tgt_overlap_policy: %s", policy.c_str());
        TGLError("propagating PyErr from handle_tgt_overlaps");
    }

    // Per-chrom event sweep: emit one row per non-empty inter-event segment.
    // auto_* picks a single winner per multi-active segment + merges adjacent
    // contiguous slices. agg emits every active interval per multi-active
    // segment + never merges.
    std::vector<ChainRow> resolved;
    resolved.reserve(rows.size() * 2);

    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i + 1;
        while (j < rows.size() && rows[j].chromid == rows[i].chromid) ++j;

        std::vector<TgtEvent> events;
        events.reserve((j - i) * 2);
        for (size_t k = i; k < j; ++k) {
            // Skip zero-length intervals; they contribute nothing to the
            // breakpoint set + must not participate in the active set (see
            // Python _handle_tgt_overlaps_auto: zero-length rows are
            // implicitly dropped by the np.unique/covers construction).
            if (rows[k].start >= rows[k].end) continue;
            events.push_back({ rows[k].start, true,  k });
            events.push_back({ rows[k].end,   false, k });
        }
        std::sort(events.begin(), events.end());

        std::set<size_t> active;
        size_t e = 0;
        while (e < events.size()) {
            int64_t pos = events[e].pos;
            while (e < events.size() && events[e].pos == pos && !events[e].is_start) {
                active.erase(events[e].idx);
                ++e;
            }
            while (e < events.size() && events[e].pos == pos && events[e].is_start) {
                active.insert(events[e].idx);
                ++e;
            }
            if (e >= events.size()) break;
            int64_t next_pos = events[e].pos;
            if (next_pos <= pos || active.empty()) continue;

            if (active.size() == 1) {
                // Single-active segment. agg never merges; auto merges.
                const bool allow_merge = !is_agg;
                append_slice(resolved, rows[*active.begin()], pos, next_pos, allow_merge);
                continue;
            }

            if (is_agg) {
                // Emit one slice per active interval. No merging.
                for (size_t active_idx : active) {
                    append_slice(resolved, rows[active_idx], pos, next_pos, /*allow_merge=*/false);
                }
                continue;
            }

            // auto_* path
            size_t winner;
            if (policy == "auto_score") winner = pick_by_score(rows, active);
            else if (policy == "auto_longer") winner = pick_by_length(rows, active);
            else /* auto_first */ winner = pick_first(rows, active);
            append_slice(resolved, rows[winner], pos, next_pos, /*allow_merge=*/true);
        }
        i = j;
    }

    rows.assign(resolved.begin(), resolved.end());
    return;
}

} // namespace

PyObject *pm_chain_intervals_resolve(PyObject *self, PyObject *args)
{
    PyObject *df_dict;
    const char *src_policy_str;
    const char *tgt_policy_str;
    if (!PyArg_ParseTuple(args, "Oss", &df_dict, &src_policy_str, &tgt_policy_str)) {
        return nullptr;
    }
    if (!PyDict_Check(df_dict)) {
        PyErr_SetString(PyExc_TypeError, "chain_df_dict must be a dict");
        return nullptr;
    }
    if (!valid_src_policy(src_policy_str)) {
        PyErr_Format(PyExc_ValueError, "Unknown src_overlap_policy: %s", src_policy_str);
        return nullptr;
    }
    if (!valid_tgt_policy(tgt_policy_str)) {
        PyErr_Format(PyExc_ValueError, "Unknown tgt_overlap_policy: %s", tgt_policy_str);
        return nullptr;
    }

    try {

    std::vector<ChainRow> rows;
    ChromInterner tgt_chrom;
    ChromInterner src_chrom;
    if (!dict_to_rows(df_dict, rows, tgt_chrom, src_chrom)) {
        return nullptr;
    }

    std::string src_policy = src_policy_str;
    std::string tgt_policy = tgt_policy_str;
    if (tgt_policy == "auto") tgt_policy = "auto_score";

    handle_src_overlaps(rows, src_policy, tgt_chrom, src_chrom);
    handle_tgt_overlaps(rows, tgt_policy, tgt_chrom, src_chrom);

    PMPY result = rows_to_dict(rows, tgt_chrom, src_chrom);
    if (!result) return nullptr;
    return_py(result);

    } catch (TGLException &e) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, e.msg());
        }
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    }
}

// =====================================================================
// G1.P3.B.2: pm_map_intervals — per-src-interval lift over a chain.
// =====================================================================

namespace {

bool valid_cluster_strategy(const char *p)
{
    return !strcmp(p, "") || !strcmp(p, "none") ||
           !strcmp(p, "union") || !strcmp(p, "sum") || !strcmp(p, "max");
}

// Parse the src_df_dict into parallel arrays. value_col_name may be empty
// to skip value-col parsing. The arrays are kept alive via PMPY wrappers
// stored in out_keepalive (returned to caller, who must release on cleanup).
struct SrcArrays {
    npy_intp n = 0;
    PyObject **chrom = nullptr;     // NPY_OBJECT
    const int64_t *start = nullptr;
    const int64_t *end = nullptr;
    const double *value = nullptr;  // nullptr when no value_col

    PMPY arr_chrom;
    PMPY arr_start;
    PMPY arr_end;
    PMPY arr_value;
};

bool parse_src_dict(PyObject *src_df_dict, const char *value_col_name,
                    SrcArrays &out)
{
    out.arr_chrom = PMPY(get_array(src_df_dict, "chrom", NPY_OBJECT), true);
    if (!out.arr_chrom) return false;
    out.arr_start = PMPY(get_array(src_df_dict, "start", NPY_INT64), true);
    if (!out.arr_start) return false;
    out.arr_end = PMPY(get_array(src_df_dict, "end", NPY_INT64), true);
    if (!out.arr_end) return false;

    out.n = PyArray_DIM((PyArrayObject *)*out.arr_chrom, 0);
    if (PyArray_DIM((PyArrayObject *)*out.arr_start, 0) != out.n ||
        PyArray_DIM((PyArrayObject *)*out.arr_end, 0) != out.n) {
        PyErr_SetString(PyExc_ValueError, "src_df_dict columns have mismatched lengths");
        return false;
    }

    out.chrom = (PyObject **)PyArray_DATA((PyArrayObject *)*out.arr_chrom);
    out.start = (const int64_t *)PyArray_DATA((PyArrayObject *)*out.arr_start);
    out.end   = (const int64_t *)PyArray_DATA((PyArrayObject *)*out.arr_end);

    if (value_col_name && value_col_name[0] != '\0') {
        PyObject *item = PyDict_GetItemString(src_df_dict, value_col_name);
        if (!item) {
            PyErr_Format(PyExc_ValueError, "value_col '%s' not found in src_df_dict", value_col_name);
            return false;
        }
        out.arr_value = PMPY(PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS), true);
        if (!out.arr_value) return false;
        if (PyArray_NDIM((PyArrayObject *)*out.arr_value) != 1) {
            PyErr_SetString(PyExc_ValueError, "value_col must be 1-D");
            return false;
        }
        if (PyArray_DIM((PyArrayObject *)*out.arr_value, 0) != out.n) {
            PyErr_SetString(PyExc_ValueError, "value_col length does not match src_df_dict");
            return false;
        }
        out.value = (const double *)PyArray_DATA((PyArrayObject *)*out.arr_value);
    }
    return true;
}

// Output column accumulators. Per-row push. Resized once at the end.
struct MappedOut {
    std::vector<int>     chromid;       // tgt chromid (interned via tgt_chrom_out)
    std::vector<int64_t> start;
    std::vector<int64_t> end;
    std::vector<int64_t> intervalID;
    std::vector<int64_t> chain_id;
    std::vector<int64_t> src_start;     // common_start; emitted as __src_start
    std::vector<int64_t> src_end;       // common_end;   emitted as __src_end
    std::vector<double>  score;         // optional (include_metadata)
    std::vector<double>  value;         // optional (value_col != "")
};

// Build the output dict. Returns a borrowed reference; caller wraps in PMPY.
PMPY mapped_to_dict(const MappedOut &m,
                    const ChromInterner &tgt_chrom,
                    bool include_metadata,
                    bool has_value,
                    const std::string &value_col_name)
{
    npy_intp n = (npy_intp)m.start.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_intervalID(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_chain_id(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_src_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_src_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    if (!py_chrom || !py_start || !py_end || !py_intervalID ||
        !py_chain_id || !py_src_start || !py_src_end) return PMPY();

    PMPY py_score;
    if (include_metadata) {
        py_score = PMPY(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
        if (!py_score) return PMPY();
    }
    PMPY py_value;
    if (has_value) {
        py_value = PMPY(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
        if (!py_value) return PMPY();
    }

    PyObject **chrom_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    for (npy_intp i = 0; i < n; ++i) chrom_out[i] = nullptr;

    int64_t *start_out      = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out        = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    int64_t *intervalID_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_intervalID);
    int64_t *chain_id_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_chain_id);
    int64_t *src_start_out  = (int64_t *)PyArray_DATA((PyArrayObject *)*py_src_start);
    int64_t *src_end_out    = (int64_t *)PyArray_DATA((PyArrayObject *)*py_src_end);
    double  *score_out      = include_metadata ? (double *)PyArray_DATA((PyArrayObject *)*py_score) : nullptr;
    double  *value_out      = has_value        ? (double *)PyArray_DATA((PyArrayObject *)*py_value) : nullptr;

    // Pre-build one PyUnicode per interned chromid. The output array holds
    // INCREFed references rather than a fresh allocation per row. On a 549k-row
    // output with ~few unique chroms this drops per-row chrom-cell cost from
    // PyUnicode_FromStringAndSize (allocates) to Py_INCREF (pointer bump).
    int max_chromid = -1;
    for (size_t i = 0; i < (size_t)n; ++i)
        if (m.chromid[i] > max_chromid) max_chromid = m.chromid[i];

    std::vector<PyObject *> chrom_cache((size_t)std::max(0, max_chromid + 1), nullptr);
    for (size_t i = 0; i < (size_t)n; ++i) {
        const int cid = m.chromid[i];
        if (chrom_cache[(size_t)cid] != nullptr) continue;
        const std::string &name = tgt_chrom.name_of(cid);
        PyObject *s = PyUnicode_FromStringAndSize(name.data(), name.size());
        if (!s) {
            for (PyObject *o : chrom_cache) Py_XDECREF(o);
            for (npy_intp j = 0; j < n; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
            }
            return PMPY();
        }
        chrom_cache[(size_t)cid] = s;
    }

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s = chrom_cache[(size_t)m.chromid[(size_t)i]];
        Py_INCREF(s);
        chrom_out[i] = s;

        start_out[i]      = m.start[(size_t)i];
        end_out[i]        = m.end[(size_t)i];
        intervalID_out[i] = m.intervalID[(size_t)i];
        chain_id_out[i]   = m.chain_id[(size_t)i];
        src_start_out[i]  = m.src_start[(size_t)i];
        src_end_out[i]    = m.src_end[(size_t)i];
        if (score_out) score_out[i] = m.score[(size_t)i];
        if (value_out) value_out[i] = m.value[(size_t)i];
    }

    for (PyObject *o : chrom_cache) Py_XDECREF(o);

    PMPY out(PyDict_New(), true);
    if (!out) return PMPY();
    PyDict_SetItemString(out, "chrom",       py_chrom);
    PyDict_SetItemString(out, "start",       py_start);
    PyDict_SetItemString(out, "end",         py_end);
    PyDict_SetItemString(out, "intervalID",  py_intervalID);
    PyDict_SetItemString(out, "chain_id",    py_chain_id);
    PyDict_SetItemString(out, "__src_start", py_src_start);
    PyDict_SetItemString(out, "__src_end",   py_src_end);
    if (include_metadata) PyDict_SetItemString(out, "score", py_score);
    if (has_value)        PyDict_SetItemString(out, value_col_name.c_str(), py_value);
    return out;
}

// Build the output dict from a MappedOutput (public type with materialized
// string chroms). Returns a PMPY-owned new reference.
PMPY mapped_to_dict_v2(const MappedOutput &m,
                        bool include_metadata,
                        bool has_value,
                        const std::string &value_col_name)
{
    npy_intp n = (npy_intp)m.start.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_intervalID(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_chain_id(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_src_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_src_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    if (!py_chrom || !py_start || !py_end || !py_intervalID ||
        !py_chain_id || !py_src_start || !py_src_end) return PMPY();

    PMPY py_score;
    if (include_metadata) {
        py_score = PMPY(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
        if (!py_score) return PMPY();
    }
    PMPY py_value;
    if (has_value) {
        py_value = PMPY(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
        if (!py_value) return PMPY();
    }

    PyObject **chrom_out  = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    for (npy_intp i = 0; i < n; ++i) chrom_out[i] = nullptr;

    int64_t *start_out      = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out        = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    int64_t *intervalID_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_intervalID);
    int64_t *chain_id_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_chain_id);
    int64_t *src_start_out  = (int64_t *)PyArray_DATA((PyArrayObject *)*py_src_start);
    int64_t *src_end_out    = (int64_t *)PyArray_DATA((PyArrayObject *)*py_src_end);
    double  *score_out      = include_metadata ? (double *)PyArray_DATA((PyArrayObject *)*py_score) : nullptr;
    double  *value_out      = has_value        ? (double *)PyArray_DATA((PyArrayObject *)*py_value) : nullptr;

    // Build one PyUnicode per unique chrom name. Reuse by pointer comparison.
    std::unordered_map<std::string, PyObject *> chrom_cache;
    bool cache_ok = true;
    for (npy_intp i = 0; i < n && cache_ok; ++i) {
        const std::string &name = m.chrom[(size_t)i];
        auto it = chrom_cache.find(name);
        PyObject *s;
        if (it == chrom_cache.end()) {
            s = PyUnicode_FromStringAndSize(name.data(), (Py_ssize_t)name.size());
            if (!s) { cache_ok = false; break; }
            chrom_cache.emplace(name, s);
        } else {
            s = it->second;
        }
        Py_INCREF(s);
        chrom_out[i] = s;

        start_out[i]      = m.start[(size_t)i];
        end_out[i]        = m.end[(size_t)i];
        intervalID_out[i] = m.intervalID[(size_t)i];
        chain_id_out[i]   = m.chain_id[(size_t)i];
        src_start_out[i]  = m.src_start[(size_t)i];
        src_end_out[i]    = m.src_end[(size_t)i];
        if (score_out) score_out[i] = m.score[(size_t)i];
        if (value_out) value_out[i] = m.value[(size_t)i];
    }

    for (auto &kv : chrom_cache) Py_XDECREF(kv.second);

    if (!cache_ok) {
        // Fill remaining chrom cells with None to keep the object array valid.
        for (npy_intp j = 0; j < n; ++j) {
            if (!chrom_out[j]) { Py_INCREF(Py_None); chrom_out[j] = Py_None; }
        }
        return PMPY();
    }

    PMPY dict(PyDict_New(), true);
    if (!dict) return PMPY();
    PyDict_SetItemString(dict, "chrom",        py_chrom);
    PyDict_SetItemString(dict, "start",        py_start);
    PyDict_SetItemString(dict, "end",          py_end);
    PyDict_SetItemString(dict, "intervalID",   py_intervalID);
    PyDict_SetItemString(dict, "chain_id",     py_chain_id);
    PyDict_SetItemString(dict, "__src_start",  py_src_start);
    PyDict_SetItemString(dict, "__src_end",    py_src_end);
    if (include_metadata) PyDict_SetItemString(dict, "score", py_score);
    if (has_value)        PyDict_SetItemString(dict, value_col_name.c_str(), py_value);
    return dict;
}

// Per-call src-aux index: pmax_end_src[i] = max(end_src[k]) for k in
// [chrom_first[chromid_src(i)], i]. Plus per-src-chrom slice bounds.
struct SrcAux {
    std::vector<int64_t> pmax_end_src;             // size n
    std::unordered_map<int, std::pair<size_t,size_t>> chrom_range; // chromid_src -> [first, last_excl)
};

void build_src_aux(const std::vector<ChainRow> &rows, SrcAux &aux)
{
    const size_t n = rows.size();
    aux.pmax_end_src.assign(n, std::numeric_limits<int64_t>::min());
    aux.chrom_range.clear();
    if (n == 0) return;

    size_t i = 0;
    while (i < n) {
        const int chrom = rows[i].chromid_src;
        const size_t first = i;
        int64_t pmax = std::numeric_limits<int64_t>::min();
        while (i < n && rows[i].chromid_src == chrom) {
            pmax = std::max(pmax, rows[i].end_src);
            aux.pmax_end_src[i] = pmax;
            ++i;
        }
        aux.chrom_range[chrom] = std::make_pair(first, i);
    }
}

// For one src interval, find every overlapping chain row and emit a
// (tgt_chrom, tgt_start, tgt_end, chain_id, common_start, common_end, score)
// candidate. Mirrors R's ChainIntervals::map_interval + add2tgt
// (rdbinterval.cpp:1122-1239) and Python's per-chrom inner loop in
// _map_intervals_vectorized.
//
// Returns the candidates as a vector for the caller to either flush
// directly to `out` or pass through cluster_resolve.
struct Candidate {
    int     chromid;        // tgt
    int64_t tgt_start;
    int64_t tgt_end;
    int64_t chain_id;
    int64_t src_start;      // common_start
    int64_t src_end;        // common_end
    double  score;
};

void map_one_src_interval(
    const std::vector<ChainRow> &rows,
    const SrcAux &aux,
    int      src_chromid,
    int64_t  src_start,
    int64_t  src_end,
    std::vector<Candidate> &out)
{
    out.clear();
    if (src_start >= src_end) return;

    auto it = aux.chrom_range.find(src_chromid);
    if (it == aux.chrom_range.end()) return;

    const size_t first = it->second.first;
    const size_t lastEx = it->second.second;

    // Upper bound: first row with start_src >= src_end (no need to look further).
    auto upper = std::lower_bound(
        rows.begin() + (ptrdiff_t)first,
        rows.begin() + (ptrdiff_t)lastEx,
        src_end,
        [](const ChainRow &r, int64_t qend) { return r.start_src < qend; });
    const size_t upper_idx = (size_t)(upper - rows.begin());

    // Lower bound: first row (in [first, upper_idx)) with pmax_end_src > src_start.
    // Binary-search the prefix-max array over [first, upper_idx).
    if (upper_idx <= first) return;
    size_t lo = first, hi = upper_idx;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (aux.pmax_end_src[mid] > src_start) hi = mid;
        else lo = mid + 1;
    }
    const size_t lower_idx = lo;
    if (lower_idx >= upper_idx) return;

    for (size_t k = lower_idx; k < upper_idx; ++k) {
        const ChainRow &r = rows[k];
        if (r.end_src <= src_start) continue;
        if (r.start_src >= src_end) break;
        // Computed overlap
        const int64_t common_start = std::max(r.start_src, src_start);
        const int64_t common_end   = std::min(r.end_src,   src_end);
        if (common_start >= common_end) continue;

        // Project to tgt coords.
        int64_t tgt_start, tgt_end;
        if (r.strand == 0) {
            // Positive strand
            const int64_t offset_start = common_start - r.start_src;
            const int64_t offset_end   = common_end   - r.start_src;
            tgt_start = r.start + offset_start;
            tgt_end   = r.start + offset_end;
        } else {
            // Negative strand
            const int64_t offset_start = common_start - r.start_src;
            const int64_t offset_end   = common_end   - r.start_src;
            tgt_start = r.end - offset_end;
            tgt_end   = r.end - offset_start;
        }

        Candidate c;
        c.chromid    = r.chromid;
        c.tgt_start  = tgt_start;
        c.tgt_end    = tgt_end;
        c.chain_id   = r.chain_id;
        c.src_start  = common_start;
        c.src_end    = common_end;
        c.score      = r.score;
        out.push_back(c);
    }
}

enum class ClusterStrategy { None, Union, Sum, Max };

ClusterStrategy parse_cluster_strategy(const char *s)
{
    if (!s || s[0] == '\0') return ClusterStrategy::None;
    if (!strcmp(s, "union")) return ClusterStrategy::Union;
    if (!strcmp(s, "sum"))   return ClusterStrategy::Sum;
    if (!strcmp(s, "max"))   return ClusterStrategy::Max;
    return ClusterStrategy::None;
}

// Apply cluster-strategy to a vector of candidates from one src interval.
// Mutates `cands` in place to keep only the surviving cluster's members.
// Mirrors R IntervalsLiftover.cpp:193-322.
void cluster_resolve(std::vector<Candidate> &cands, ClusterStrategy strat)
{
    const size_t n = cands.size();
    if (strat == ClusterStrategy::None || n <= 1) return;

    // Union-find
    std::vector<size_t> parent(n);
    std::vector<size_t> rank_uf(n, 0);
    for (size_t i = 0; i < n; ++i) parent[i] = i;

    std::function<size_t(size_t)> find_root = [&](size_t x) -> size_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](size_t a, size_t b) {
        size_t ra = find_root(a), rb = find_root(b);
        if (ra == rb) return;
        if (rank_uf[ra] < rank_uf[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_uf[ra] == rank_uf[rb]) rank_uf[ra]++;
    };

    // (a) Union by chain_id
    std::unordered_map<int64_t, size_t> first_for_chain;
    for (size_t i = 0; i < n; ++i) {
        auto it = first_for_chain.find(cands[i].chain_id);
        if (it != first_for_chain.end()) unite(i, it->second);
        else first_for_chain[cands[i].chain_id] = i;
    }

    // (b) Union by source overlap (sweep)
    std::vector<size_t> sorted_idx(n);
    for (size_t i = 0; i < n; ++i) sorted_idx[i] = i;
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](size_t a, size_t b) {
        return cands[a].src_start < cands[b].src_start;
    });
    int64_t max_end = std::numeric_limits<int64_t>::min();
    size_t  max_end_idx = 0;
    for (size_t k = 0; k < n; ++k) {
        size_t i = sorted_idx[k];
        if (cands[i].src_start < max_end) unite(i, max_end_idx);
        if (cands[i].src_end > max_end) {
            max_end = cands[i].src_end;
            max_end_idx = i;
        }
    }

    // Group by root + compute mass
    std::unordered_map<size_t, std::vector<size_t>> components;
    for (size_t i = 0; i < n; ++i)
        components[find_root(i)].push_back(i);

    std::vector<size_t> best_indices;
    int64_t best_mass = -1;
    int64_t best_min_start = std::numeric_limits<int64_t>::max();

    for (auto &kv : components) {
        std::vector<size_t> &comp = kv.second;
        // Sort by src_start within the component
        std::sort(comp.begin(), comp.end(), [&](size_t a, size_t b) {
            return cands[a].src_start < cands[b].src_start;
        });

        int64_t mass = 0;
        const int64_t min_start = cands[comp[0]].src_start;

        if (strat == ClusterStrategy::Union) {
            int64_t union_end = std::numeric_limits<int64_t>::min();
            for (size_t idx : comp) {
                const int64_t s = cands[idx].src_start;
                const int64_t e = cands[idx].src_end;
                if (e > union_end) {
                    mass += e - std::max(s, union_end);
                    union_end = e;
                }
            }
        } else if (strat == ClusterStrategy::Sum) {
            for (size_t idx : comp)
                mass += cands[idx].src_end - cands[idx].src_start;
        } else { // Max
            for (size_t idx : comp) {
                int64_t len = cands[idx].src_end - cands[idx].src_start;
                if (len > mass) mass = len;
            }
        }

        if (mass > best_mass ||
            (mass == best_mass && min_start < best_min_start)) {
            best_mass = mass;
            best_min_start = min_start;
            best_indices = comp;
        }
    }

    // Preserve original input order within the surviving cluster.
    std::sort(best_indices.begin(), best_indices.end());

    std::vector<Candidate> kept;
    kept.reserve(best_indices.size());
    for (size_t idx : best_indices) kept.push_back(cands[idx]);
    cands.swap(kept);
}

} // namespace

// -------------------------------------------------------------------------
// Public C++ API (declared in PMChainIntervals.h)
// -------------------------------------------------------------------------

ClusterStrat parse_cluster_strategy_str(const std::string &s)
{
    if (s.empty() || s == "none") return ClusterStrat::NONE;
    if (s == "union") return ClusterStrat::UNION;
    if (s == "sum")   return ClusterStrat::SUM;
    if (s == "max")   return ClusterStrat::MAX;
    throw std::invalid_argument(
        "cluster_strategy must be '', 'none', 'union', 'sum', or 'max', got '" + s + "'");
}

// Helper: map ClusterStrat -> internal ClusterStrategy enum used by
// the anonymous-namespace cluster_resolve().
static ClusterStrategy to_internal_strat(ClusterStrat s)
{
    switch (s) {
        case ClusterStrat::UNION: return ClusterStrategy::Union;
        case ClusterStrat::SUM:   return ClusterStrategy::Sum;
        case ClusterStrat::MAX:   return ClusterStrategy::Max;
        default:                  return ClusterStrategy::None;
    }
}

void map_intervals_cpp(
    const SrcRowInput &src,
    const ChainRowInput &chain,
    bool include_metadata,
    ClusterStrat cluster_strategy,
    MappedOutput &out)
{
    // Build local interners from the chain POD arrays.
    ChromInterner tgt_chrom;
    ChromInterner src_chrom;
    std::vector<ChainRow> chain_rows;
    chain_rows.reserve((size_t)chain.n);
    for (int64_t i = 0; i < chain.n; ++i) {
        ChainRow r;
        r.chromid     = tgt_chrom.intern(chain.chrom[i]);
        r.start       = chain.start[i];
        r.end         = chain.end[i];
        r.strand      = chain.strand[i];
        r.chromid_src = src_chrom.intern(chain.chromsrc[i]);
        r.start_src   = chain.startsrc[i];
        r.end_src     = chain.endsrc[i];
        r.strand_src  = chain.strandsrc[i];
        r.chain_id    = chain.chain_id[i];
        r.score       = chain.score[i];
        chain_rows.push_back(r);
    }

    sort_by_src(chain_rows);
    SrcAux aux;
    build_src_aux(chain_rows, aux);

    const ClusterStrategy strat = to_internal_strat(cluster_strategy);
    const bool has_value = (src.value != nullptr);

    // Reserve using src.n as a capacity hint.
    out.chrom.reserve((size_t)src.n);
    out.start.reserve((size_t)src.n);
    out.end.reserve((size_t)src.n);
    out.intervalID.reserve((size_t)src.n);
    out.chain_id.reserve((size_t)src.n);
    out.src_start.reserve((size_t)src.n);
    out.src_end.reserve((size_t)src.n);
    if (include_metadata) out.score.reserve((size_t)src.n);
    if (has_value)        out.value.reserve((size_t)src.n);

    std::vector<Candidate> cands;
    for (int64_t i = 0; i < src.n; ++i) {
        const std::string &ckey = src.chrom[i];
        int chk = src_chrom.find_existing(ckey);
        if (chk < 0) continue;

        map_one_src_interval(chain_rows, aux, chk, src.start[i], src.end[i], cands);
        if (cands.empty()) continue;

        cluster_resolve(cands, strat);

        for (const Candidate &c : cands) {
            out.chrom.push_back(tgt_chrom.name_of(c.chromid));
            out.start.push_back(c.tgt_start);
            out.end.push_back(c.tgt_end);
            out.intervalID.push_back(i);
            out.chain_id.push_back(c.chain_id);
            out.src_start.push_back(c.src_start);
            out.src_end.push_back(c.src_end);
            if (include_metadata) out.score.push_back(c.score);
            if (has_value)        out.value.push_back(src.value[i]);
        }
    }
}

// -------------------------------------------------------------------------
// Python entry point: thin wrapper around map_intervals_cpp
// -------------------------------------------------------------------------

PyObject *pm_map_intervals(PyObject *self, PyObject *args)
{
    PyObject *src_df_dict;
    PyObject *chain_df_dict;
    const char *value_col_name;
    int include_metadata_flag;
    const char *cluster_strategy_str;

    if (!PyArg_ParseTuple(args, "OOsps",
                          &src_df_dict, &chain_df_dict,
                          &value_col_name, &include_metadata_flag,
                          &cluster_strategy_str)) {
        return nullptr;
    }
    if (!PyDict_Check(src_df_dict)) {
        PyErr_SetString(PyExc_TypeError, "src_df_dict must be a dict");
        return nullptr;
    }
    if (!PyDict_Check(chain_df_dict)) {
        PyErr_SetString(PyExc_TypeError, "chain_df_dict must be a dict");
        return nullptr;
    }
    if (!valid_cluster_strategy(cluster_strategy_str)) {
        PyErr_Format(PyExc_ValueError,
                     "cluster_strategy must be '', 'union', 'sum', or 'max', got '%s'",
                     cluster_strategy_str);
        return nullptr;
    }
    const bool include_metadata = include_metadata_flag != 0;
    const bool has_value = (value_col_name && value_col_name[0] != '\0');
    const std::string value_col_str = has_value ? std::string(value_col_name) : std::string();

    try {

    // --- Parse src dict into SrcArrays (keepalive in src object) ---
    SrcArrays src_arrs;
    if (!parse_src_dict(src_df_dict, value_col_name, src_arrs)) return nullptr;

    // Materialise src chrom PyObject** -> std::vector<std::string>
    std::vector<std::string> src_chrom_strs;
    src_chrom_strs.reserve((size_t)src_arrs.n);
    for (npy_intp i = 0; i < src_arrs.n; ++i) {
        if (!src_arrs.chrom[i] || !PyUnicode_Check(src_arrs.chrom[i])) {
            PyErr_Format(PyExc_TypeError,
                         "src_df_dict['chrom'][%lld] is not a string", (long long)i);
            return nullptr;
        }
        Py_ssize_t clen = 0;
        const char *cstr = PyUnicode_AsUTF8AndSize(src_arrs.chrom[i], &clen);
        if (!cstr) return nullptr;
        src_chrom_strs.emplace_back(cstr, (size_t)clen);
    }

    // --- Parse chain dict into parallel POD vectors ---
    PMPY arr_chrom(get_array(chain_df_dict, "chrom",     NPY_OBJECT), true);
    if (!arr_chrom) return nullptr;
    PMPY arr_chromsrc(get_array(chain_df_dict, "chromsrc",  NPY_OBJECT), true);
    if (!arr_chromsrc) return nullptr;
    PMPY arr_start(get_array(chain_df_dict, "start",     NPY_INT64), true);
    if (!arr_start) return nullptr;
    PMPY arr_end(get_array(chain_df_dict, "end",       NPY_INT64), true);
    if (!arr_end) return nullptr;
    PMPY arr_strand(get_array(chain_df_dict, "strand",    NPY_INT64), true);
    if (!arr_strand) return nullptr;
    PMPY arr_startsrc(get_array(chain_df_dict, "startsrc",  NPY_INT64), true);
    if (!arr_startsrc) return nullptr;
    PMPY arr_endsrc(get_array(chain_df_dict, "endsrc",    NPY_INT64), true);
    if (!arr_endsrc) return nullptr;
    PMPY arr_strandsrc(get_array(chain_df_dict, "strandsrc", NPY_INT64), true);
    if (!arr_strandsrc) return nullptr;
    PMPY arr_chain_id(get_array(chain_df_dict, "chain_id",  NPY_INT64), true);
    if (!arr_chain_id) return nullptr;
    PMPY arr_score(get_array(chain_df_dict, "score",     NPY_DOUBLE), true);
    if (!arr_score) return nullptr;

    const npy_intp n_chain = PyArray_DIM((PyArrayObject *)*arr_chrom, 0);
    PyArrayObject *checks[] = {
        (PyArrayObject *)*arr_chromsrc, (PyArrayObject *)*arr_start,
        (PyArrayObject *)*arr_end,      (PyArrayObject *)*arr_strand,
        (PyArrayObject *)*arr_startsrc, (PyArrayObject *)*arr_endsrc,
        (PyArrayObject *)*arr_strandsrc,(PyArrayObject *)*arr_chain_id,
        (PyArrayObject *)*arr_score,
    };
    for (PyArrayObject *a : checks) {
        if (PyArray_DIM(a, 0) != n_chain) {
            PyErr_SetString(PyExc_ValueError, "chain_df_dict columns have mismatched lengths");
            return nullptr;
        }
    }

    PyObject **chrom_in    = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chrom);
    PyObject **chromsrc_in = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chromsrc);

    std::vector<std::string> chain_chrom_strs;
    std::vector<std::string> chain_chromsrc_strs;
    chain_chrom_strs.reserve((size_t)n_chain);
    chain_chromsrc_strs.reserve((size_t)n_chain);
    for (npy_intp i = 0; i < n_chain; ++i) {
        if (!chrom_in[i] || !PyUnicode_Check(chrom_in[i])) {
            PyErr_Format(PyExc_TypeError,
                         "chain_df_dict['chrom'][%lld] is not a string", (long long)i);
            return nullptr;
        }
        if (!chromsrc_in[i] || !PyUnicode_Check(chromsrc_in[i])) {
            PyErr_Format(PyExc_TypeError,
                         "chain_df_dict['chromsrc'][%lld] is not a string", (long long)i);
            return nullptr;
        }
        Py_ssize_t clen = 0;
        const char *cp = PyUnicode_AsUTF8AndSize(chrom_in[i], &clen);
        if (!cp) return nullptr;
        chain_chrom_strs.emplace_back(cp, (size_t)clen);

        Py_ssize_t cslen = 0;
        const char *csp = PyUnicode_AsUTF8AndSize(chromsrc_in[i], &cslen);
        if (!csp) return nullptr;
        chain_chromsrc_strs.emplace_back(csp, (size_t)cslen);
    }

    const int64_t *chain_start     = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_start);
    const int64_t *chain_end       = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_end);
    const int64_t *chain_strand    = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_strand);
    const int64_t *chain_startsrc  = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_startsrc);
    const int64_t *chain_endsrc    = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_endsrc);
    const int64_t *chain_strandsrc = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_strandsrc);
    const int64_t *chain_chain_id  = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_chain_id);
    const double  *chain_score     = (const double  *)PyArray_DATA((PyArrayObject *)*arr_score);

    SrcRowInput src_in{
        src_chrom_strs.empty() ? nullptr : src_chrom_strs.data(),
        src_arrs.start,
        src_arrs.end,
        src_arrs.value,
        (int64_t)src_arrs.n
    };
    ChainRowInput chain_in{
        chain_chrom_strs.empty()    ? nullptr : chain_chrom_strs.data(),
        chain_start,
        chain_end,
        chain_strand,
        chain_chromsrc_strs.empty() ? nullptr : chain_chromsrc_strs.data(),
        chain_startsrc,
        chain_endsrc,
        chain_strandsrc,
        chain_chain_id,
        chain_score,
        (int64_t)n_chain
    };

    ClusterStrat strat = parse_cluster_strategy_str(std::string(cluster_strategy_str));

    MappedOutput out;
    map_intervals_cpp(src_in, chain_in, include_metadata, strat, out);

    PMPY result = mapped_to_dict_v2(out, include_metadata, has_value, value_col_str);
    if (!result) return nullptr;
    return_py(result);

    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (TGLException &e) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, e.msg());
        }
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    }
}
