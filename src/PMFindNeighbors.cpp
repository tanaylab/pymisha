/*
 * PMFindNeighbors.cpp
 *
 * Adapted from R misha's GenomeTrackFindNeighbors.cpp for pymisha
 * Implements efficient nearest neighbor search for genomic intervals
 */

#include <cstdint>
#include <climits>
#include <algorithm>
#include <vector>
#include <set>
#include <limits>
#include <utility>

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDb.h"
#include "GInterval.h"
#include "SegmentFinder.h"

using namespace std;

// Result structure for 1D interval neighbors. Neighbors are ordered by query, then by
// distance, then by the neighbor's coordinates, so which neighbors are chosen among those at the
// same distance, and their order, does not depend on the input order (as R misha 5.17). Only
// neighbors with the same coordinates keep their input order (by row).
struct IntervNeighbor {
    int64_t id1;    // Query interval index
    int64_t id2;    // Target interval index (-1 if not found)
    int64_t dist;   // Distance between intervals
    int64_t start2; // Target coordinates
    int64_t end2;

    IntervNeighbor(int64_t _id1, int64_t _id2, int64_t _dist, int64_t _start2 = 0, int64_t _end2 = 0)
        : id1(_id1), id2(_id2), dist(_dist), start2(_start2), end2(_end2) {}

    bool operator<(const IntervNeighbor &o) const {
        if (id1 != o.id1)
            return id1 < o.id1;
        if (llabs(dist) != llabs(o.dist))
            return llabs(dist) < llabs(o.dist);
        if (start2 != o.start2)
            return start2 < o.start2;
        if (end2 != o.end2)
            return end2 < o.end2;
        return id2 < o.id2;
    }
};

// The intervals2 of one chromosome at distance 0 from a query - those that overlap or touch it,
// start <= query end and end >= query start - in the order neighbors are kept: (start, end, row).
// A query that overlaps K of them would otherwise walk all K in the nearest-neighbor iterator to
// choose the maxneighbors with the smallest coordinates; here each one kept costs O(log n), through
// a tree of the largest end over the sorted intervals. (R misha's ZeroDistNeighbors.)
class ZeroDistNeighbors {
public:
    struct Item {
        int64_t start;
        int64_t end;
        int64_t row;
        bool operator<(const Item &o) const {
            return start < o.start || (start == o.start && (end < o.end || (end == o.end && row < o.row)));
        }
    };

    void clear() { m_items.clear(); m_maxend.clear(); m_cap = 0; }
    bool empty() const { return m_items.empty() && !m_cap; }

    void build(vector<Item> &&items) {
        m_items = std::move(items);
        sort(m_items.begin(), m_items.end());
        m_cap = 1;
        while (m_cap < m_items.size())
            m_cap *= 2;
        m_maxend.assign(2 * m_cap, numeric_limits<int64_t>::min());
        for (size_t i = 0; i < m_items.size(); ++i)
            m_maxend[m_cap + i] = m_items[i].end;
        for (size_t i = m_cap - 1; i > 0; --i)
            m_maxend[i] = max(m_maxend[2 * i], m_maxend[2 * i + 1]);
    }

    // Calls take(item) for the first n intervals at distance 0 from [qstart, qend), in order
    template <class F> void first(int64_t qstart, int64_t qend, int n, F take) const {
        for (size_t i = find(1, 0, m_cap, 0, qstart); n > 0 && i < m_items.size() && m_items[i].start <= qend; i = find(1, 0, m_cap, i + 1, qstart), --n)
            take(m_items[i]);
    }

private:
    vector<Item>    m_items;
    vector<int64_t> m_maxend;
    size_t          m_cap = 0;

    // the first index >= from whose end is >= qstart, or the number of items
    size_t find(size_t node, size_t lo, size_t hi, size_t from, int64_t qstart) const {
        if (hi <= from || m_maxend[node] < qstart)
            return m_items.size();
        if (hi - lo == 1)
            return lo;
        size_t mid = lo + (hi - lo) / 2;
        size_t res = find(2 * node, lo, mid, from, qstart);
        return res < m_items.size() ? res : find(2 * node + 1, mid, hi, from, qstart);
    }
};

// Helper: Parse intervals from PMDataFrame into GInterval vector
// Returns vector of intervals with original indices stored in udata
// Also optionally returns strand values
static void parse_intervals(PMDataFrame &df,
                           const GenomeChromKey &chromkey,
                           vector<GInterval> &intervals,
                           vector<int> *strands = nullptr) {
    size_t nrows = df.num_rows();
    intervals.reserve(nrows);
    if (strands) strands->reserve(nrows);

    // Find column indices
    int chrom_col = -1, start_col = -1, end_col = -1, strand_col = -1;
    for (size_t i = 0; i < df.num_cols(); ++i) {
        const char *name = df.col_name(i);
        if (strcmp(name, "chrom") == 0) chrom_col = i;
        else if (strcmp(name, "start") == 0) start_col = i;
        else if (strcmp(name, "end") == 0) end_col = i;
        else if (strcmp(name, "strand") == 0) strand_col = i;
    }

    if (chrom_col < 0 || start_col < 0 || end_col < 0) {
        TGLError("Intervals must have chrom, start, end columns");
    }

    for (size_t i = 0; i < nrows; ++i) {
        const char *chrom = df.val_str(i, chrom_col);
        int64_t start = df.val_long(i, start_col);
        int64_t end = df.val_long(i, end_col);

        int chromid = chromkey.chrom2id(chrom);

        // Store the strand column on the GInterval itself (default 0 = no
        // strand, matching R's convert_rintervs). dist2interv uses the
        // *target* interval's strand to sign the distance: strand 0 yields an
        // unsigned (positive) distance, strand +/-1 yields a signed one. Not
        // setting it here would make every distance unsigned, so a signed
        // (e.g. all-negative) distance window would never match.
        char gi_strand = 0;
        if (strand_col >= 0) {
            gi_strand = (char)df.val_long(i, strand_col);
        }

        GInterval interv(chromid, start, end, gi_strand, GInterval::cast2udata(i));
        intervals.push_back(interv);

        if (strands) {
            // Separate query-strand vector for the use_intervals1_strand path,
            // which (like R's get_strand_value) defaults a missing/zero strand
            // to + (1) rather than 0.
            int strand = 1;  // default to + strand
            if (strand_col >= 0) {
                strand = (int)df.val_long(i, strand_col);
                if (strand == 0) strand = 1;  // treat 0 as +
            }
            strands->push_back(strand);
        }
    }
}

// Helper: Sort intervals by chromosome and start
static void sort_intervals(vector<GInterval> &intervals) {
    // stable: equal intervals keep their input order on every platform, so
    // neighbor ties don't depend on the C++ library
    stable_sort(intervals.begin(), intervals.end());
}

// Helper: Get original index from interval
static inline int64_t get_orig_idx(const GInterval &interv) {
    int64_t idx;
    memcpy(&idx, &interv.udata, sizeof(idx));
    return idx;
}

// Main neighbor-finding function
PyObject *pm_find_neighbors(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha;

        PyObject *py_intervs1;
        PyObject *py_intervs2;
        int maxneighbors;
        double mindist, maxdist;
        int na_if_notfound;
        int use_intervals1_strand = 0;  // default: use intervals2 strand

        if (!PyArg_ParseTuple(args, "OOiddi|i",
                              &py_intervs1, &py_intervs2,
                              &maxneighbors, &mindist, &maxdist,
                              &na_if_notfound, &use_intervals1_strand)) {
            return_err();
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            TGLError("Database not initialized. Call gdb_init() first.");
        }

        if (maxneighbors < 1) {
            TGLError("maxneighbors must be >= 1");
        }

        // Clamp to a bound past any genome: (int64_t)inf is undefined (INT64_MIN on
        // x86, so an infinite maxdist matched nothing) and llabs(INT64_MIN) overflows.
        mindist = std::max(-1e18, std::min(1e18, mindist));
        maxdist = std::max(-1e18, std::min(1e18, maxdist));
        if (mindist > maxdist) {
            TGLError("mindist must be <= maxdist");
        }

        // Intervals are already converted to pymisha format by Python wrapper
        PMPY pm_df1(py_intervs1);
        PMPY pm_df2(py_intervs2);

        PMDataFrame df1(pm_df1, "intervals1");
        PMDataFrame df2(pm_df2, "intervals2");

        size_t nrows1 = df1.num_rows();
        size_t nrows2 = df2.num_rows();

        // Handle empty inputs
        if (nrows1 == 0 || (nrows2 == 0 && !na_if_notfound)) {
            return_none();
        }

        // Parse intervals
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        vector<GInterval> intervals1, intervals2;
        vector<int> strands1;

        // If use_intervals1_strand is set, extract strand info from intervals1
        if (use_intervals1_strand) {
            parse_intervals(df1, chromkey, intervals1, &strands1);
        } else {
            parse_intervals(df1, chromkey, intervals1);
        }
        parse_intervals(df2, chromkey, intervals2);

        // Sort intervals
        sort_intervals(intervals1);
        sort_intervals(intervals2);

        // Compute absolute distance bounds
        int64_t abs_maxdist = max(llabs((int64_t)mindist), llabs((int64_t)maxdist));
        int64_t abs_mindist = min(llabs((int64_t)mindist), llabs((int64_t)maxdist));

        // Handle signed distance range
        bool only_positive = (mindist > 0 && maxdist > 0);
        bool only_negative = (mindist < 0 && maxdist < 0);

        // Result storage
        vector<IntervNeighbor> results;

        // Segment finder for nearest neighbor queries
        SegmentFinder<GInterval> segment_finder;
        SegmentFinder<GInterval>::NNIterator nn_iter(&segment_finder);

        int cur_chromid = -1;
        // Single-pass index over the sorted intervals2 vector. Each chrom
        // transition just advances this pointer forward to the next
        // matching chromid run, so total insertion work is O(|intervals2|)
        // instead of O(|intervals1_chrom_changes| * |intervals2|).
        size_t intervals2_pos = 0;
        size_t chrom_run_start = 0;  // the intervals2 of cur_chromid: [chrom_run_start, intervals2_pos)
        ZeroDistNeighbors zero_dist;  // built for cur_chromid when a query first needs it

        // Process each query interval
        for (size_t i = 0; i < intervals1.size(); ++i) {
            const GInterval &query = intervals1[i];

            // Check for new chromosome - need to rebuild segment finder
            if (query.chromid != cur_chromid) {
                cur_chromid = query.chromid;

                // Reset segment finder for this chromosome
                uint64_t chrom_size = chromkey.get_chrom_size(cur_chromid);
                segment_finder.reset(0, chrom_size);

                // Both intervals1 and intervals2 are sorted by chromid, so
                // we only need to skip past earlier chroms in intervals2
                // and then insert the contiguous run for cur_chromid.
                while (intervals2_pos < intervals2.size() &&
                       intervals2[intervals2_pos].chromid < cur_chromid) {
                    ++intervals2_pos;
                }
                chrom_run_start = intervals2_pos;
                while (intervals2_pos < intervals2.size() &&
                       intervals2[intervals2_pos].chromid == cur_chromid) {
                    segment_finder.insert(intervals2[intervals2_pos]);
                    ++intervals2_pos;
                }
                zero_dist.clear();
            }

            // Set up excluded area if needed (for mindist filtering)
            Segment excluded_area(0, -1);  // Invalid segment = no exclusion
            if (only_positive || only_negative) {
                excluded_area = Segment(query.start - abs_mindist + 1,
                                        query.end + abs_mindist - 1);
            }

            // Begin nearest neighbor iteration
            nn_iter.begin(query, excluded_area);

            int num_neighbors = 0;
            int64_t orig_idx1 = get_orig_idx(query);
            int64_t cutoff = 0;  // |dist| of the maxneighbors-th neighbor found
            size_t first = results.size();
            bool zero_dist_ties = false;

            while (!nn_iter.is_end()) {
                const GInterval &target = *nn_iter;

                // Compute signed distance
                int64_t dist;
                if (use_intervals1_strand) {
                    // Use query strand to determine distance directionality
                    int query_strand = strands1[orig_idx1];
                    dist = query.dist2interv_with_query_strand(target, query_strand);
                } else {
                    // Default: use target strand for directionality
                    dist = query.dist2interv(target);
                }

                // Check if we've gone past the max distance
                if (llabs(dist) > abs_maxdist) {
                    break;
                }
                // The iterator yields the neighbors by |dist|. Once maxneighbors are found, those
                // at the distance of the last one are taken too, and the ties are cut below.
                if (num_neighbors >= maxneighbors && llabs(dist) > cutoff)
                    break;

                // Check distance range (signed)
                if (dist >= (int64_t)mindist && dist <= (int64_t)maxdist) {
                    results.push_back(IntervNeighbor(orig_idx1, get_orig_idx(target), dist, target.start, target.end));
                    if (++num_neighbors == maxneighbors)
                        cutoff = llabs(dist);
                    // the rows kept: tied candidates beyond maxneighbors are cut below
                    pymisha.verify_max_data_size(first + min(num_neighbors, maxneighbors), "neighbors result");
                    // maxneighbors at distance 0: the ties can be all the intervals2 that overlap
                    // the query, taken below in their order instead of walked here
                    if (num_neighbors == maxneighbors && cutoff == 0) {
                        zero_dist_ties = true;
                        break;
                    }
                }

                nn_iter.next();
            }

            // Distance 0 is in [mindist, maxdist] here (it was reached), so every interval at
            // distance 0 is a candidate, and the ones kept are the first maxneighbors in
            // (start, end, row) order. Their distance is 0 with any strand.
            if (zero_dist_ties) {
                if (zero_dist.empty()) {
                    vector<ZeroDistNeighbors::Item> items;
                    items.reserve(intervals2_pos - chrom_run_start);
                    for (size_t k = chrom_run_start; k < intervals2_pos; ++k)
                        items.push_back({intervals2[k].start, intervals2[k].end, get_orig_idx(intervals2[k])});
                    zero_dist.build(std::move(items));
                }
                results.erase(results.begin() + first, results.end());
                zero_dist.first(query.start, query.end, maxneighbors, [&](const ZeroDistNeighbors::Item &item) {
                    results.push_back(IntervNeighbor(orig_idx1, item.row, 0, item.start, item.end));
                    pymisha.verify_max_data_size(results.size(), "neighbors result");
                });
            }

            // neighbors tied at the cutoff distance: keep those with the smallest coordinates
            if (num_neighbors > maxneighbors) {
                sort(results.begin() + first, results.end());
                results.erase(results.begin() + first + maxneighbors, results.end());
                num_neighbors = maxneighbors;
            }

            // Handle na_if_notfound
            if (na_if_notfound && num_neighbors == 0) {
                results.push_back(IntervNeighbor(orig_idx1, -1, 0));
                pymisha.verify_max_data_size(results.size(), "neighbors result");
            }

            check_interrupt();
        }

        if (results.empty()) {
            return_none();
        }

        // Sort results by query, then distance, then the target coordinates, then target
        sort(results.begin(), results.end());

        // Build output DataFrame
        // Columns: all from intervals1 + all from intervals2 (renamed only if collision) + dist
        size_t ncols1 = df1.num_cols();
        size_t ncols2 = df2.num_cols();
        size_t ncols_out = ncols1 + ncols2 + 1;  // +1 for dist

        PMDataFrame out_df(results.size(), ncols_out, "neighbors_result");

        // Store column types from source DataFrames
        vector<PMDataFrame::Type> types1(ncols1);
        vector<PMDataFrame::Type> types2(ncols2);

        // Collect column names from df1 to detect collisions
        set<string> df1_colnames;
        for (size_t c = 0; c < ncols1; ++c) {
            df1_colnames.insert(df1.col_name(c));
        }

        // Initialize columns from intervals1 (use actual types from source)
        for (size_t c = 0; c < ncols1; ++c) {
            const char *name = df1.col_name(c);
            types1[c] = df1.col_type(c);
            out_df.init_col(c, name, types1[c]);
        }

        // Initialize columns from intervals2
        // Only add '1' suffix if the name collides with df1 column names
        for (size_t c = 0; c < ncols2; ++c) {
            const char *orig_name = df2.col_name(c);
            string new_name;
            if (df1_colnames.count(orig_name) > 0) {
                // Collision - add '1' suffix
                new_name = string(orig_name) + "1";
            } else {
                // No collision - keep original name
                new_name = orig_name;
            }
            types2[c] = df2.col_type(c);
            out_df.init_col(ncols1 + c, new_name.c_str(), types2[c]);
        }

        // Initialize dist column as DOUBLE to support NaN for na_if_notfound case
        out_df.init_col(ncols1 + ncols2, "dist", PMDataFrame::DOUBLE);

        // Fill data
        for (size_t r = 0; r < results.size(); ++r) {
            const IntervNeighbor &res = results[r];

            // Copy query interval data
            for (size_t c = 0; c < ncols1; ++c) {
                switch (types1[c]) {
                    case PMDataFrame::STR:
                        out_df.val_str(r, c, df1.val_str(res.id1, c));
                        break;
                    case PMDataFrame::LONG:
                        out_df.val_long(r, c, df1.val_long(res.id1, c));
                        break;
                    case PMDataFrame::DOUBLE:
                        out_df.val_double(r, c, df1.val_double(res.id1, c));
                        break;
                    case PMDataFrame::BOOL:
                        out_df.val_bool(r, c, df1.val_bool(res.id1, c));
                        break;
                    default:
                        out_df.val_double(r, c, df1.val_double(res.id1, c));
                }
            }

            // Copy target interval data (or NA if not found)
            if (res.id2 >= 0) {
                for (size_t c = 0; c < ncols2; ++c) {
                    switch (types2[c]) {
                        case PMDataFrame::STR:
                            out_df.val_str(r, ncols1 + c, df2.val_str(res.id2, c));
                            break;
                        case PMDataFrame::LONG:
                            out_df.val_long(r, ncols1 + c, df2.val_long(res.id2, c));
                            break;
                        case PMDataFrame::DOUBLE:
                            out_df.val_double(r, ncols1 + c, df2.val_double(res.id2, c));
                            break;
                        case PMDataFrame::BOOL:
                            out_df.val_bool(r, ncols1 + c, df2.val_bool(res.id2, c));
                            break;
                        default:
                            out_df.val_double(r, ncols1 + c, df2.val_double(res.id2, c));
                    }
                }
                out_df.val_double(r, ncols1 + ncols2, (double)res.dist);
            } else {
                // NA values for target columns
                for (size_t c = 0; c < ncols2; ++c) {
                    switch (types2[c]) {
                        case PMDataFrame::STR:
                            out_df.val_str(r, ncols1 + c, NULL);
                            break;
                        case PMDataFrame::LONG:
                            // For LONG, use -1 as NA sentinel (not ideal but consistent)
                            out_df.val_long(r, ncols1 + c, -1);
                            break;
                        case PMDataFrame::DOUBLE:
                            out_df.val_double(r, ncols1 + c, NPY_NAN);
                            break;
                        case PMDataFrame::BOOL:
                            out_df.val_bool(r, ncols1 + c, false);
                            break;
                        default:
                            out_df.val_double(r, ncols1 + c, NPY_NAN);
                    }
                }
                // NA distance
                out_df.val_double(r, ncols1 + ncols2, NPY_NAN);
            }
        }

        return_py(out_df.construct_py(true));

    } catch (TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return_err();
    } catch (const bad_alloc &e) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return_err();
    }

    return_none();
}
