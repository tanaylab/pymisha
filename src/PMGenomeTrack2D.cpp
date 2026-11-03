/*
 * PMGenomeTrack2D.cpp
 */

#include "PMGenomeTrack2D.h"

#include <cstring>
#include <sys/stat.h>

#include "GenomeChromKey.h"
#include "PMDb.h"
#include "TGLException.h"
#include "TrackIndex2D.h"

// g_pmdb is declared `extern` in PMDb.h; no need to redeclare here.

// GenomeTrack::FORMAT_SIGNATURES of COMPUTED
static const int32_t SIGNATURE_COMPUTED = -11;

PMGenomeTrack2D::PMGenomeTrack2D() = default;

PMGenomeTrack2D::~PMGenomeTrack2D() { close_pair(); }

void PMGenomeTrack2D::init(const std::string &track_name)
{
    if (!g_pmdb || !g_pmdb->is_initialized()) {
        TGLError("Database not initialized");
    }
    if (!g_pmdb->track_exists(track_name)) {
        TGLError("Track does not exist: %s", track_name.c_str());
    }

    m_track_name = track_name;
    m_track_path = g_pmdb->track_path(track_name);

    // Load the indexed-format 2D index once, if present. It's cached globally
    // by TrackIndex2D::get_track_index_2d so this is cheap on repeat calls.
    m_index = TrackIndex2D::get_track_index_2d(m_track_path);

    // Determine track type. For per-pair-file tracks, GenomeTrack::get_type
    // reads the per-pair file signatures and returns RECTS / POINTS. For
    // 2D-indexed tracks, GenomeTrack::get_type misidentifies them as SPARSE
    // (it only knows the 1D MISHATDX magic), so we fall back to the 2D
    // index's own track-type field.
    if (m_index && m_index->is_loaded()) {
        MishaTrack2DType idx_type = m_index->get_track_type();
        if (idx_type == MishaTrack2DType::RECTS) {
            m_is_points = false;
        } else if (idx_type == MishaTrack2DType::POINTS) {
            m_is_points = true;
        } else {
            TGLError("2D scanner: track '%s' indexed 2D type %u not supported",
                     track_name.c_str(), static_cast<unsigned>(idx_type));
        }
        return;
    }

    GenomeTrack::Type track_type = GenomeTrack::get_type(
        m_track_path.c_str(), g_pmdb->chromkey(), false);

    if (track_type == GenomeTrack::RECTS) {
        m_is_points = false;
    } else if (track_type == GenomeTrack::COMPUTED) {
        // Its objects are stored as RECTS objects are; only their enumeration
        // (query_objects, for a 2D iterator) is meaningful, not the statistics,
        // which R recomputes through the track's Computer2D.
        m_is_points = false;
        m_is_computed = true;
    } else if (track_type == GenomeTrack::POINTS) {
        m_is_points = true;
    } else {
        TGLError("2D scanner: track '%s' has type '%s'; only RECTS and POINTS "
                 "are supported in this session",
                 track_name.c_str(), GenomeTrack::TYPE_NAMES[track_type]);
    }
}

bool PMGenomeTrack2D::set_chrom_pair(int chromid1, int chromid2)
{
    // Same pair, already resolved? Short-circuit.
    if (m_lookup_state != LOOKUP_NOT_ATTEMPTED &&
        m_cur_chromid1 == chromid1 && m_cur_chromid2 == chromid2) {
        if (m_lookup_state == LOOKUP_OPEN_FAILED) {
            // The previous attempt on this pair failed to open the file.
            // Don't paper over - re-raise so the caller can surface the
            // problem instead of silently returning "no data".
            TGLError("2D scanner: pair (%d,%d) for track '%s' previously failed to open",
                     chromid1, chromid2, m_track_name.c_str());
        }
        return m_lookup_state == LOOKUP_FOUND && m_num_objs > 0;
    }

    // Different pair (or first call) - release prior state.
    close_pair();
    m_cur_chromid1 = chromid1;
    m_cur_chromid2 = chromid2;

    // Indexed format takes precedence.
    if (m_index && m_index->is_loaded()) {
        const Track2DPairEntry *entry = m_index->get_entry(
            static_cast<uint32_t>(chromid1), static_cast<uint32_t>(chromid2));
        if (entry == nullptr || entry->length == 0) {
            // No entry in the index = genuinely absent (cached).
            m_lookup_state = LOOKUP_ABSENT;
            return false;
        }
        if (!try_open_indexed(*entry)) {
            // Entry existed but the dat slice would not parse.
            m_lookup_state = LOOKUP_OPEN_FAILED;
            TGLError("2D scanner: pair (%d,%d) for track '%s' has corrupt indexed data",
                     chromid1, chromid2, m_track_name.c_str());
            return false;  // unreachable; TGLError throws
        }
        m_lookup_state = LOOKUP_FOUND;
        return m_num_objs > 0;
    }

    // Per-pair file fallback.
    int open_result = try_open_per_pair(chromid1, chromid2);
    if (open_result == 0) {
        // No per-pair file exists - genuinely absent.
        m_lookup_state = LOOKUP_ABSENT;
        return false;
    }
    if (open_result < 0) {
        // File exists but failed to mmap / parse - hard error.
        m_lookup_state = LOOKUP_OPEN_FAILED;
        TGLError("2D scanner: pair (%d,%d) for track '%s' exists but failed to open",
                 chromid1, chromid2, m_track_name.c_str());
        return false;  // unreachable
    }
    m_lookup_state = LOOKUP_FOUND;
    return m_num_objs > 0;
}

bool PMGenomeTrack2D::try_open_indexed(const Track2DPairEntry &entry)
{
    // Open the track.dat mmap if we don't already have it. The mmap
    // outlives any single set_chrom_pair() call so subsequent pair
    // switches just rebase m_cur_buf into the same mapping.
    if (m_mmap.size() == 0) {
        std::string dat_path = m_track_path + "/track.dat";
        if (!m_mmap.open(dat_path, false)) {
            return false;
        }
    }
    if (entry.offset + entry.length > m_mmap.size()) {
        return false;
    }

    const uint8_t *slice = m_mmap.data() + entry.offset;
    return parse_pair_header(slice, static_cast<size_t>(entry.length));
}

int PMGenomeTrack2D::try_open_per_pair(int chromid1, int chromid2)
{
    // The file of the pair as every reader finds it (GenomeTrack::ChromFiles): named by the
    // canonical names, as R misha names it, or by aliases, as pymisha 0.14 and earlier named
    // it in a per-chromosome database (by the chrom_sizes.txt names)
    const GenomeTrack::ChromFiles &files = g_pmdb->chrom_files(m_track_path);
    if (!files.exists(chromid1, chromid2))
        return 0;  // genuinely absent
    std::string path = files.path(chromid1, chromid2);

    // File exists. Try to open it. Failure here is a hard error.
    if (!m_mmap.open(path, false)) {
        return -1;
    }
    if (!parse_pair_header(m_mmap.data(), m_mmap.size())) {
        m_mmap.close();
        return -1;
    }
    return 1;
}

bool PMGenomeTrack2D::parse_pair_header(const uint8_t *buf, size_t len)
{
    if (len < 12) return false;

    int32_t signature;
    std::memcpy(&signature, buf, sizeof(int32_t));
    bool sig_is_points;
    size_t header = 4;  // the signature
    if (signature == quadtree::SIGNATURE_RECTS) {
        sig_is_points = false;
    } else if (signature == quadtree::SIGNATURE_POINTS) {
        sig_is_points = true;
    } else if (signature == SIGNATURE_COMPUTED && m_is_computed && len >= 20) {
        // The Computer2D header: its type, then its state. The area and test
        // computers (types 0 and 3) have no state; the others are not read.
        int32_t computer_type;
        std::memcpy(&computer_type, buf + 4, sizeof(int32_t));
        if (computer_type != 0 && computer_type != 3)
            TGLError("2D scanner: track '%s' uses a Computer2D of type %d, which is not supported",
                     m_track_name.c_str(), computer_type);
        sig_is_points = false;
        header = 8;
    } else {
        return false;
    }
    // Sanity: the per-pair signature must match the track-attribute type.
    if (sig_is_points != m_is_points) {
        TGLError("2D scanner: track '%s' attribute type disagrees with "
                 "on-disk signature for pair (%d,%d)",
                 m_track_name.c_str(), m_cur_chromid1, m_cur_chromid2);
    }

    std::memcpy(&m_num_objs, buf + header, sizeof(uint64_t));
    if (m_num_objs == 0) {
        m_root_chunk_fpos = 0;
    } else {
        std::memcpy(&m_root_chunk_fpos, buf + header + 8, sizeof(int64_t));
    }
    m_cur_buf = buf;
    m_cur_len = len;
    return true;
}

quadtree::BatchQueryStats PMGenomeTrack2D::query_stats_batch(
    const int64_t *rects, size_t n,
    const quadtree::DiagonalBand *band) const
{
    quadtree::BatchQueryStats batch;
    if (m_lookup_state != LOOKUP_FOUND || m_num_objs == 0 || n == 0) {
        batch.resize(n);
        return batch;
    }
    return quadtree::query_stats_batch(
        reinterpret_cast<const char *>(m_cur_buf), m_cur_len,
        m_is_points, m_root_chunk_fpos,
        rects, n, band);
}

quadtree::QueryObjects PMGenomeTrack2D::query_objects(
    int64_t qx1, int64_t qy1, int64_t qx2, int64_t qy2,
    const quadtree::DiagonalBand *band) const
{
    quadtree::QueryObjects result;
    if (m_lookup_state != LOOKUP_FOUND || m_num_objs == 0) {
        return result;
    }
    return quadtree::query_objects(
        reinterpret_cast<const char *>(m_cur_buf), m_cur_len,
        m_is_points, m_root_chunk_fpos,
        qx1, qy1, qx2, qy2, band);
}

void PMGenomeTrack2D::close_pair()
{
    m_mmap.close();
    m_cur_buf = nullptr;
    m_cur_len = 0;
    m_num_objs = 0;
    m_root_chunk_fpos = 0;
    m_lookup_state = LOOKUP_NOT_ATTEMPTED;
}
