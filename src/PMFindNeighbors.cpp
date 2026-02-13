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

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDb.h"
#include "GInterval.h"
#include "SegmentFinder.h"

using namespace std;

// Result structure for 1D interval neighbors
struct IntervNeighbor {
    int64_t id1;    // Query interval index
    int64_t id2;    // Target interval index (-1 if not found)
    int64_t dist;   // Distance between intervals

    IntervNeighbor(int64_t _id1, int64_t _id2, int64_t _dist)
        : id1(_id1), id2(_id2), dist(_dist) {}

    bool operator<(const IntervNeighbor &o) const {
        // Sort by query id first, then by absolute distance, then by target id
        return id1 < o.id1 ||
               (id1 == o.id1 && llabs(dist) < llabs(o.dist)) ||
               (id1 == o.id1 && llabs(dist) == llabs(o.dist) && id2 < o.id2);
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

        GInterval interv(chromid, start, end, 0, GInterval::cast2udata(i));
        intervals.push_back(interv);

        if (strands) {
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
    sort(intervals.begin(), intervals.end());
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

        // Process each query interval
        for (size_t i = 0; i < intervals1.size(); ++i) {
            const GInterval &query = intervals1[i];

            // Check for new chromosome - need to rebuild segment finder
            if (query.chromid != cur_chromid) {
                cur_chromid = query.chromid;

                // Reset segment finder for this chromosome
                uint64_t chrom_size = chromkey.get_chrom_size(cur_chromid);
                segment_finder.reset(0, chrom_size);

                // Insert all intervals2 on this chromosome
                for (size_t j = 0; j < intervals2.size(); ++j) {
                    if (intervals2[j].chromid == cur_chromid) {
                        segment_finder.insert(intervals2[j]);
                    }
                }
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

                // Check distance range (signed)
                if (dist >= (int64_t)mindist && dist <= (int64_t)maxdist) {
                    int64_t orig_idx2 = get_orig_idx(target);
                    results.push_back(IntervNeighbor(orig_idx1, orig_idx2, dist));

                    pymisha.verify_max_data_size(results.size(), "neighbors result");

                    num_neighbors++;
                    if (num_neighbors >= maxneighbors) {
                        break;
                    }
                }

                nn_iter.next();
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

        // Sort results by query, then distance, then target
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
