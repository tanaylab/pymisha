/*
 * PMGenomeTrack2D.h
 *
 * Wrapper that opens a 2D track for one chrom-pair at a time and forwards
 * stats / object queries to the quadtree namespace. Holds at most one open
 * mmap; switching pairs releases the previous mapping.
 *
 * Supports three storage layouts (in priority order):
 *   1. Indexed: track.dat + track.idx (single concatenated file).
 *   2. Per-pair file with pymisha name (e.g. "1-2").
 *   3. Per-pair file with R misha name (e.g. "chr1-chr2").
 *
 * Note on indexed pair access: TrackIndex2D gives us (offset, length)
 * into track.dat. We compute root_chunk_fpos relative to the start of
 * the pair slice (a bytes() copy), and pass that buffer to the quadtree
 * query functions.
 */

#ifndef PMGENOMETRACK2D_H_
#define PMGENOMETRACK2D_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "GenomeTrack.h"
#include "MmapFile.h"
#include "QuadTreeReader.h"

class TrackIndex2D;
struct Track2DPairEntry;

class PMGenomeTrack2D {
public:
    // Pair-lookup result for the most recent set_chrom_pair() call.
    enum LookupState {
        LOOKUP_NOT_ATTEMPTED, // set_chrom_pair() has not yet been called for this pair
        LOOKUP_FOUND,         // pair was opened; num_objs may be 0 (empty pair) or >0
        LOOKUP_ABSENT,        // pair is genuinely missing from the track (no index entry, no per-pair file)
        LOOKUP_OPEN_FAILED,   // a candidate file existed but failed to open / parse - HARD ERROR for the next caller
    };

    PMGenomeTrack2D();
    ~PMGenomeTrack2D();

    // Initialize the wrapper for a given physical track name.
    // Throws TGLException on:
    //   - unknown track,
    //   - non-2D track,
    //   - non-RECTS/POINTS track type.
    void init(const std::string &track_name);

    // Switch to a different chrom-pair. Releases the previous mapping.
    // Returns true if the pair has data (num_objs > 0), false otherwise.
    // After a false return all query_* calls are no-ops.
    bool set_chrom_pair(int chromid1, int chromid2);

    // Whether the underlying track stores POINTS (vs RECTS).
    bool is_points() const { return m_is_points; }

    // Number of objects in the currently-open pair (0 if no pair open).
    uint64_t num_objs() const { return m_num_objs; }

    // Result of the most recent set_chrom_pair() lookup.
    LookupState lookup_state() const { return m_lookup_state; }

    // Batch stats query for N rectangles on the currently-open pair.
    // `rects` is an N*4 array of int64 in (qx1, qy1, qx2, qy2) order.
    // Returns a BatchQueryStats sized N (all zeros if no pair open).
    quadtree::BatchQueryStats query_stats_batch(const int64_t *rects, size_t n,
                                                const quadtree::DiagonalBand *band) const;

    // Query objects intersecting (qx1, qy1, qx2, qy2) on the currently-open pair.
    // Returns an empty QueryObjects if no pair is open or num_objs == 0.
    quadtree::QueryObjects query_objects(int64_t qx1, int64_t qy1,
                                         int64_t qx2, int64_t qy2,
                                         const quadtree::DiagonalBand *band) const;

    // Track name (for diagnostics).
    const std::string &track_name() const { return m_track_name; }

private:
    std::string                  m_track_name;
    std::string                  m_track_path;
    bool                         m_is_points{false};

    // Indexed-format index (may be nullptr if the track is per-pair).
    std::shared_ptr<TrackIndex2D> m_index;

    // The mmap that backs m_cur_buf. For the indexed format this is a single
    // mmap of track.dat that lives for the lifetime of the PMGenomeTrack2D
    // (or until close_pair() is called). For per-pair format it is a mmap
    // of that one file and gets reopened on each pair switch.
    MmapFile m_mmap;

    // Currently-open pair state.
    const uint8_t *m_cur_buf{nullptr};
    size_t         m_cur_len{0};
    uint64_t       m_num_objs{0};
    int64_t        m_root_chunk_fpos{0};
    int            m_cur_chromid1{-1};
    int            m_cur_chromid2{-1};
    LookupState    m_lookup_state{LOOKUP_NOT_ATTEMPTED};

    // Try opening the pair from the indexed format given a resolved entry.
    // Returns true on success.
    bool try_open_indexed(const Track2DPairEntry &entry);

    // Try opening the pair from a per-pair file.
    // Returns 0 if no per-pair file exists, 1 if it opened OK,
    // -1 if a candidate file existed but failed to mmap / parse.
    int try_open_per_pair(int chromid1, int chromid2);

    // Parse the (signature, num_objs, root_chunk_fpos) header that lives
    // at the start of every pair buffer.
    // Returns true on success, false if signature is unrecognized.
    bool parse_pair_header(const uint8_t *buf, size_t len);

    // Release any open pair.
    void close_pair();
};

#endif /* PMGENOMETRACK2D_H_ */
