// pm_liftover_track_2d: gtrack_liftover of a 2D (RECTS or POINTS) source track.
//
// A port of the 2D path of R misha 5.12.1's gtrack_liftover (GTrackLiftover.cpp),
// written so that the track files are the bytes R writes:
//
//  1. Every object of the source is mapped through the chain, both ends, and
//     appended to a buffer file per TARGET chromosome pair in the new track's
//     directory: 20-byte (x, y, value) records for a POINTS source, 36-byte
//     (x1, x2, y1, y2, value) records for RECTS. The source is read one
//     chromosome pair at a time with a bounded chunk cache.
//  2. All the buffers are flushed and closed, then each pair is reopened and
//     written in turn, and its buffer removed once it is read:
//     - POINTS (write_2d_points): split into subtrees by CONFIG['max_data_size'],
//       as R's gtrack.2d.import_contacts splits a pair; each subtree is sorted by
//       (x, y, read order), points on one target point are merged by
//       multi_target_agg, and the points go to a PointsQuadTreeCachedSerializer.
//       Only one subtree is in memory at a time.
//     - RECTS: overlapping rects are cut into disjoint cells (aggregate_2d_rects)
//       and written as one RectsQuadTree.
//
// The quad trees are R's own code: StatQuadTree.h, StatQuadTreeCached.h and
// StatQuadTreeCachedSerializer.h are shared with R misha unchanged.

#include "pymisha.h"

#include <cstdint>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferedFile.h"
#include "PMLiftoverAggregate.h"
#include "PMSourceTrack.h"
#include "GenomeChromKey.h"
#include "GenomeTrack.h"
#include "StatQuadTreeCached.h"
#include "StatQuadTreeCachedSerializer.h"
#include "TGLException.h"
#include "TrackIndex2D.h"

namespace {

// R misha's gtrack.chunk.size and gtrack.num.chunks defaults (ConfigurationDefaults.h):
// the chunk size the quad trees are written with, and no limit on the chunks held.
const int64_t TRACK_CHUNK_SIZE = 100000;
const int64_t TRACK_NUM_CHUNKS = 0;

// The source is read in one pass, so it needs only the chunks on the iterator's stack;
// R bounds its chunk cache to 100 chunks (gtrack.num.chunks = 0 would cache them all).
const int64_t SRC_NUM_CHUNKS = 100;

// GenomeTrack::FORMAT_SIGNATURES of RECTS and POINTS
const int32_t RECTS_SIGNATURE = -9;
const int32_t POINTS_SIGNATURE = -10;

struct AggregationConfig {
    AggType type;
    bool    na_rm;
    int64_t min_n;      // -1: none
    int64_t nth_index;  // 1-based, NTH only
};

//------------------------------------------ chain ------------------------------------------

// One block of the chain, as R's ChainInterval (strands: 0 = +, 1 = -).
struct ChainInterval {
    int     chromid;       // target chromosome: its id in the target database
    int64_t start;
    int64_t end;
    int64_t strand;
    int     chromid_src;   // source chromosome: its index in src_id2chrom
    int64_t start_src;
    int64_t end_src;
    int64_t chain_id;
    double  score;
};

// A target interval a source interval maps to
struct TgtInterval {
    int     chromid;
    int64_t start;
    int64_t end;
};

// The chain sorted by source, mapping a source interval to its target intervals as R's
// ChainIntervals::map_interval does (with the whole chain as the hint).
class ChainIntervals {
public:
    void init(std::vector<ChainInterval> &&intervs, int num_src_chroms, const std::string &tgt_overlap_policy);

    // tgt_intervs is cleared
    void map_interval(int chromid_src, int64_t start, int64_t end, std::vector<TgtInterval> &tgt_intervs);

private:
    struct Mapped {
        TgtInterval interval;
        double      score;
        int64_t     chain_id;
        int64_t     span;
    };

    std::vector<ChainInterval> m_intervs;
    std::vector<int64_t>       m_pmax_end_src;     // prefix-max of end_src within each source chromosome
    std::vector<size_t>        m_chrom_first;
    std::vector<size_t>        m_chrom_last_excl;
    bool                       m_auto_score{false};
    std::vector<Mapped>        m_mapped;
    std::vector<bool>          m_keep;
};

void ChainIntervals::init(std::vector<ChainInterval> &&intervs, int num_src_chroms, const std::string &tgt_overlap_policy)
{
    m_intervs = std::move(intervs);
    m_auto_score = tgt_overlap_policy == "auto_score" || tgt_overlap_policy == "auto";

    // R's ChainInterval::SrcCompare
    std::sort(m_intervs.begin(), m_intervs.end(), [](const ChainInterval &a, const ChainInterval &b) {
        if (a.chromid_src != b.chromid_src)
            return a.chromid_src < b.chromid_src;
        if (a.start_src != b.start_src)
            return a.start_src < b.start_src;
        if (a.end_src != b.end_src)
            return a.end_src < b.end_src;
        if (a.chromid != b.chromid)
            return a.chromid < b.chromid;
        if (a.start != b.start)
            return a.start < b.start;
        return a.chain_id < b.chain_id;
    });

    m_chrom_first.assign(num_src_chroms, 0);
    m_chrom_last_excl.assign(num_src_chroms, 0);
    m_pmax_end_src.assign(m_intervs.size(), 0);
    for (size_t i = 0; i < m_intervs.size(); ) {
        int chromid = m_intervs[i].chromid_src;
        int64_t pmax = std::numeric_limits<int64_t>::min();
        m_chrom_first[chromid] = i;
        for (; i < m_intervs.size() && m_intervs[i].chromid_src == chromid; ++i) {
            pmax = std::max(pmax, m_intervs[i].end_src);
            m_pmax_end_src[i] = pmax;
        }
        m_chrom_last_excl[chromid] = i;
    }
}

void ChainIntervals::map_interval(int chromid_src, int64_t start, int64_t end, std::vector<TgtInterval> &tgt_intervs)
{
    tgt_intervs.clear();
    m_mapped.clear();

    size_t lo = m_chrom_first[chromid_src];
    size_t last_excl = m_chrom_last_excl[chromid_src];

    // the first chain of the chromosome that reaches past start
    size_t hi = last_excl;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (m_pmax_end_src[mid] > start)
            hi = mid;
        else
            lo = mid + 1;
    }

    for (size_t i = lo; i < last_excl && m_intervs[i].start_src < end; ++i) {
        const ChainInterval &c = m_intervs[i];
        if (c.end_src <= start)
            continue;

        int64_t common_start = std::max(c.start_src, start);
        int64_t common_end = std::min(c.end_src, end);
        Mapped m;

        m.interval.chromid = c.chromid;
        if (c.strand == 0) {
            m.interval.start = c.start + common_start - c.start_src;
            m.interval.end = c.start + common_end - c.start_src;
        } else {
            m.interval.start = c.end - (common_end - c.start_src);
            m.interval.end = c.end - (common_start - c.start_src);
        }
        m.score = c.score;
        m.chain_id = c.chain_id;
        m.span = m.interval.end - m.interval.start;
        m_mapped.push_back(m);
    }

    if (!m_auto_score) {
        for (const Mapped &m : m_mapped)
            tgt_intervs.push_back(m.interval);
        return;
    }

    // auto_score: of two target intervals that overlap keep the one with the higher score,
    // then the longer one, then the lower chain id (ChainIntervals::add2tgt)
    m_keep.assign(m_mapped.size(), true);
    for (size_t i = 0; i < m_mapped.size(); ++i) {
        if (!m_keep[i])
            continue;
        for (size_t j = i + 1; j < m_mapped.size(); ++j) {
            if (!m_keep[j])
                continue;
            const TgtInterval &a = m_mapped[i].interval;
            const TgtInterval &b = m_mapped[j].interval;
            if (a.chromid == b.chromid && a.start < b.end && b.start < a.end) {
                bool i_wins;
                if (m_mapped[i].score != m_mapped[j].score)
                    i_wins = m_mapped[i].score > m_mapped[j].score;
                else if (m_mapped[i].span != m_mapped[j].span)
                    i_wins = m_mapped[i].span > m_mapped[j].span;
                else
                    i_wins = m_mapped[i].chain_id < m_mapped[j].chain_id;
                if (i_wins)
                    m_keep[j] = false;
                else {
                    m_keep[i] = false;
                    break;
                }
            }
        }
    }
    for (size_t i = 0; i < m_mapped.size(); ++i) {
        if (m_keep[i])
            tgt_intervs.push_back(m_mapped[i].interval);
    }
}

//------------------------------------------ buffers ------------------------------------------

void write_chk(BufferedFile &file, const void *buf, uint64_t size, const char *what)
{
    if (file.write(buf, size) != size) {
        if (file.error())
            TGLError("Failed to write %s to file %s: %s", what, file.file_name().c_str(), strerror(errno));
        TGLError("Failed to write %s to file %s", what, file.file_name().c_str());
    }
}

void flush_chk(BufferedFile &file, const char *what)
{
    if (file.flush())
        TGLError("Failed to write %s to file %s: %s", what, file.file_name().c_str(), strerror(errno));
}

void open_chk(BufferedFile &file, const std::string &path, const char *mode)
{
    if (file.open(path.c_str(), mode))
        TGLError("Opening file %s: %s", path.c_str(), strerror(errno));
}

// Lifted points of a POINTS track are buffered as (x, y, val) records. In memory a point
// also carries seq, its position in the buffer, which is its contribution id: it orders
// first / last / nth and keeps the aggregation from merging distinct points.
struct PointVal2D { int64_t x, y; float v; uint32_t seq; };

const int POINT_RECORD_SIZE = 2 * sizeof(int64_t) + sizeof(float);
const int RECT_RECORD_SIZE = 4 * sizeof(int64_t) + sizeof(float);

void write_point(BufferedFile &file, int64_t x, int64_t y, float v)
{
    char buf[POINT_RECORD_SIZE];
    memcpy(buf, &x, sizeof(x));
    memcpy(buf + sizeof(x), &y, sizeof(y));
    memcpy(buf + sizeof(x) + sizeof(y), &v, sizeof(v));
    write_chk(file, buf, POINT_RECORD_SIZE, "points");
}

bool read_point(BufferedFile &file, int64_t &x, int64_t &y, float &v)
{
    char buf[POINT_RECORD_SIZE];
    uint64_t size = file.read(buf, POINT_RECORD_SIZE);
    if (size != POINT_RECORD_SIZE) {
        if (!size && file.eof())
            return false;
        if (file.error())
            TGLError("Failed to read a file %s: %s", file.file_name().c_str(), strerror(errno));
        TGLError("Invalid format of a file %s", file.file_name().c_str());
    }
    memcpy(&x, buf, sizeof(x));
    memcpy(&y, buf + sizeof(x), sizeof(y));
    memcpy(&v, buf + sizeof(x) + sizeof(y), sizeof(v));

    // an infinite source value becomes NaN, as in the 1D path
    if (std::isinf(v))
        v = std::numeric_limits<float>::quiet_NaN();
    return true;
}

// x1, x2, y1, y2, v - the order of R's BufferedIntervals2D
void write_rect(BufferedFile &file, const TgtInterval &interv1, const TgtInterval &interv2, float v)
{
    char buf[RECT_RECORD_SIZE];
    char *p = buf;
    memcpy(p, &interv1.start, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(p, &interv1.end, sizeof(int64_t));   p += sizeof(int64_t);
    memcpy(p, &interv2.start, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(p, &interv2.end, sizeof(int64_t));   p += sizeof(int64_t);
    memcpy(p, &v, sizeof(v));
    write_chk(file, buf, RECT_RECORD_SIZE, "intervals");
}

struct RectVal2D { int64_t x1, y1, x2, y2; float v; };

bool read_rect(BufferedFile &file, RectVal2D &r)
{
    char buf[RECT_RECORD_SIZE];
    uint64_t size = file.read(buf, RECT_RECORD_SIZE);
    if (size != RECT_RECORD_SIZE) {
        if (!size && file.eof())
            return false;
        if (file.error())
            TGLError("Failed to read a file %s: %s", file.file_name().c_str(), strerror(errno));
        TGLError("Invalid format of a file %s", file.file_name().c_str());
    }
    const char *p = buf;
    memcpy(&r.x1, p, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(&r.x2, p, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(&r.y1, p, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(&r.y2, p, sizeof(int64_t)); p += sizeof(int64_t);
    memcpy(&r.v, p, sizeof(r.v));

    if (std::isinf(r.v))
        r.v = std::numeric_limits<float>::quiet_NaN();
    return true;
}

// One contribution of value v whose id (start, end, chain id) is seq, covering the whole
// locus: R's aggregation_state_add(state, v, 1, 1, seq, seq, seq).
void add_contribution(std::vector<BinContribution> &state, float v, int64_t seq)
{
    BinContribution c;
    c.value = (double)v;
    c.overlap_len = 1.0;
    c.coverage_frac = 1.0;
    c.start = seq;
    c.end = seq;
    c.is_na = std::isnan(c.value);
    c.chain_id = seq;
    state.push_back(c);
}

float aggregate(const AggregationConfig &cfg, const std::vector<BinContribution> &state)
{
    double aggregated = aggregate_value_for_bin(state, cfg.type, cfg.na_rm, cfg.min_n, cfg.nth_index);
    return std::isnan(aggregated) ? std::numeric_limits<float>::quiet_NaN() : (float)aggregated;
}

void write_signature(BufferedFile &file, const std::string &path, int32_t signature)
{
    open_chk(file, path, "wb");
    write_chk(file, &signature, sizeof(signature), "a track");
}

//------------------------------------------ writing a pair ------------------------------------------

// Writes the points lifted into one target chromosome pair (buffered in infilename) as a
// POINTS track file. Points that landed on the same target point are merged by cfg. As in
// gtrack.2d.import_contacts the pair is split into subtrees by max_data_size, and only one
// subtree is held in memory at a time. The buffer file is removed once it is read: after
// the split, so that the pair is not on disk twice.
void write_2d_points(const std::string &infilename, const std::string &filename, const std::string &dirname,
                     const std::string &chrom1, const std::string &chrom2, int64_t chrom_size1, int64_t chrom_size2,
                     uint64_t max_data_size, const AggregationConfig &cfg)
{
    BufferedFile infile;
    open_chk(infile, infilename, "r");

    int64_t num_points = infile.file_size() / POINT_RECORD_SIZE;
    // below 1 it counts as 1 rather than dividing by zero
    max_data_size = std::max(max_data_size, (uint64_t)1);
    int64_t num_subtrees = std::max((int64_t)((uint64_t)num_points / max_data_size), (int64_t)1);
    num_subtrees = (int64_t)1 << (2 * (int)(log2(num_subtrees) / 2));  // round down to a power of 4

    BufferedFile outfile;
    write_signature(outfile, filename, POINTS_SIGNATURE);
    PointsQuadTreeCachedSerializer serializer;
    serializer.begin(outfile, 0, 0, chrom_size1, chrom_size2, (unsigned)num_subtrees, false, TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);

    BufferedFiles subtree_files(num_subtrees > 1 ? num_subtrees : 0);
    int64_t x, y;
    float v;

    if (num_subtrees > 1) {
        const Rectangles &subarenas = serializer.get_subarenas();
        char subtree_filename[FILENAME_MAX];

        for (int64_t i = 0; i < num_subtrees; ++i) {
            snprintf(subtree_filename, sizeof(subtree_filename), "%s/_subtree%ld", dirname.c_str(), (long)i);
            subtree_files[i] = new BufferedFile();
            open_chk(*subtree_files[i], subtree_filename, "w+");
        }

        while (read_point(infile, x, y, v)) {
            Point point(x, y);
            for (Rectangles::const_iterator isubarena = subarenas.begin(); isubarena != subarenas.end(); ++isubarena) {
                if (point.do_intersect(*isubarena)) {
                    write_point(*subtree_files[isubarena - subarenas.begin()], x, y, v);
                    break;
                }
            }
            check_interrupt();
        }

        for (int64_t i = 0; i < num_subtrees; ++i) {
            flush_chk(*subtree_files[i], "points");
            subtree_files[i]->seek(0, SEEK_SET);
        }

        infile.close();
        unlink(infilename.c_str());
    }

    std::vector<PointVal2D> points;
    std::vector<BinContribution> state;

    for (int64_t i = 0; i < num_subtrees; ++i) {
        BufferedFile &file = num_subtrees > 1 ? *subtree_files[i] : infile;

        // seq is 32-bit; check the count before reserving, or a huge subtree fails as "Out of memory"
        int64_t num_file_points = file.file_size() / POINT_RECORD_SIZE;
        if (num_file_points > (int64_t)std::numeric_limits<uint32_t>::max())
            TGLError("Too many points lifted into chromosome pair (%s, %s); lower CONFIG['max_data_size']",
                     chrom1.c_str(), chrom2.c_str());

        points.clear();
        points.reserve(num_file_points);
        while (read_point(file, x, y, v)) {
            PointVal2D p = { x, y, v, (uint32_t)points.size() };
            points.push_back(p);
            check_interrupt();
        }

        std::string read_filename = file.file_name();
        file.close();
        unlink(read_filename.c_str());

        std::sort(points.begin(), points.end(), [](const PointVal2D &a, const PointVal2D &b) {
            if (a.x != b.x)
                return a.x < b.x;
            if (a.y != b.y)
                return a.y < b.y;
            return a.seq < b.seq;
        });

        for (size_t first = 0, last; first < points.size(); first = last) {
            state.clear();
            for (last = first; last < points.size() && points[last].x == points[first].x && points[last].y == points[first].y; ++last)
                add_contribution(state, points[last].v, points[last].seq);

            serializer.insert(PointsQuadTree::ValueType(points[first].x, points[first].y, aggregate(cfg, state)));
            check_interrupt();
        }
    }

    serializer.end();
}

// Aggregates possibly-overlapping mapped rectangles into DISJOINT rectangles before
// inserting them into a StatQuadTree (which requires non-overlapping objects). Disjoint
// source rects can map onto overlapping target rects because the chain shifts x and y
// independently. The x and y boundaries are coordinate-compressed into a grid and every
// grid cell gets the aggregate (by cfg) of the values of all source rects covering it.
// Each cell carries a unique contribution id so the aggregation never folds distinct
// sources together. Cells are emitted one rect each (no run merging).
void aggregate_2d_rects(std::vector<RectVal2D> &rects, const AggregationConfig &cfg, RectsQuadTree &qtree)
{
    if (rects.empty())
        return;

    std::vector<int64_t> xs, ys;
    xs.reserve(rects.size() * 2);
    ys.reserve(rects.size() * 2);
    for (const RectVal2D &r : rects) {
        xs.push_back(r.x1); xs.push_back(r.x2);
        ys.push_back(r.y1); ys.push_back(r.y2);
    }
    std::sort(xs.begin(), xs.end()); xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end()); ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

    std::vector<size_t> by_x1(rects.size()), by_x2(rects.size());
    for (size_t i = 0; i < rects.size(); ++i) { by_x1[i] = i; by_x2[i] = i; }
    std::sort(by_x1.begin(), by_x1.end(), [&](size_t a, size_t b) { return rects[a].x1 < rects[b].x1; });
    std::sort(by_x2.begin(), by_x2.end(), [&](size_t a, size_t b) { return rects[a].x2 < rects[b].x2; });

    std::set<size_t> active;
    size_t ia = 0, ir = 0;
    int64_t encounter = 0; // unique id per contribution

    for (size_t xi = 0; xi + 1 < xs.size(); ++xi) {
        int64_t xa = xs[xi], xb = xs[xi + 1];

        while (ir < by_x2.size() && rects[by_x2[ir]].x2 <= xa) { active.erase(by_x2[ir]); ++ir; }
        while (ia < by_x1.size() && rects[by_x1[ia]].x1 <= xa) { active.insert(by_x1[ia]); ++ia; }

        if (active.empty())
            continue;

        // per y-band aggregation over the rects active in this x-slab; every active rect
        // fully covers each cell it touches, so all contributors weigh equally
        std::map<int, std::vector<BinContribution>> band_states;
        for (size_t irect : active) {
            const RectVal2D &r = rects[irect];
            int b_lo = (int)(std::lower_bound(ys.begin(), ys.end(), r.y1) - ys.begin());
            int b_hi = (int)(std::lower_bound(ys.begin(), ys.end(), r.y2) - ys.begin());
            for (int b = b_lo; b < b_hi; ++b) {
                add_contribution(band_states[b], r.v, encounter);
                ++encounter;
            }
        }

        for (const auto &kv : band_states)
            qtree.insert(RectsQuadTree::ValueType(xa, ys[kv.first], xb, ys[kv.first + 1], aggregate(cfg, kv.second)));
        check_interrupt();
    }
}

// Writes the rects lifted into one target chromosome pair (buffered in infilename) as a
// RECTS track file, cut into disjoint cells; the buffer file is removed once it is read.
void write_2d_rects(const std::string &infilename, const std::string &filename, int64_t chrom_size1, int64_t chrom_size2,
                    const AggregationConfig &cfg)
{
    std::vector<RectVal2D> rects;
    {
        BufferedFile infile;
        RectVal2D r;
        open_chk(infile, infilename, "r");
        while (read_rect(infile, r))
            rects.push_back(r);
    }
    unlink(infilename.c_str());

    RectsQuadTree qtree;
    qtree.reset(0, 0, chrom_size1, chrom_size2);
    aggregate_2d_rects(rects, cfg, qtree);

    if (!qtree.empty()) {
        BufferedFile outfile;
        write_signature(outfile, filename, RECTS_SIGNATURE);
        RectsQuadTreeCached qtree_cached(TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);
        qtree_cached.serialize(outfile, qtree);
        flush_chk(outfile, "a track");
    }
}

//------------------------------------------ reading the source ------------------------------------------

// The type of a per-pair 2D track: the signature of the first (by name) file that is a
// RECTS or POINTS file; 0 if none is.
int32_t per_pair_track_signature(const std::string &dir)
{
    std::vector<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d)
        throw std::runtime_error("Cannot list source track directory: " + dir);
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] != '.')
            names.emplace_back(e->d_name);
    }
    closedir(d);
    std::sort(names.begin(), names.end());

    for (const std::string &name : names) {
        if (name == "track.idx" || name == "track.dat")
            continue;
        std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) || !S_ISREG(st.st_mode))
            continue;
        FILE *fp = fopen(path.c_str(), "rb");
        if (!fp)
            continue;
        int32_t signature = 0;
        size_t n = fread(&signature, sizeof(signature), 1, fp);
        fclose(fp);
        if (n == 1 && (signature == RECTS_SIGNATURE || signature == POINTS_SIGNATURE))
            return signature;
    }
    return 0;
}

// The chrom id by which an indexed source is read, per chain source chromosome (-1: the
// source genome lacks it): a chain chromosome is matched in the source database by name,
// as is, then without or with the "chr" prefix.
std::vector<int> map_src_chainid2genomeid(const std::vector<std::string> &src_id2chrom, const std::vector<std::string> &src_chroms)
{
    std::unordered_map<std::string, int> chrom2id;
    for (size_t i = 0; i < src_chroms.size(); ++i)
        chrom2id.emplace(src_chroms[i], (int)i);

    std::vector<int> chainid2genomeid(src_id2chrom.size(), -1);
    for (size_t i = 0; i < src_id2chrom.size(); ++i) {
        const std::string &name = src_id2chrom[i];
        auto ichrom = chrom2id.find(name);
        if (ichrom == chrom2id.end())
            ichrom = name.compare(0, 3, "chr") ? chrom2id.find("chr" + name) : chrom2id.find(name.substr(3));
        if (ichrom != chrom2id.end())
            chainid2genomeid[i] = ichrom->second;
    }
    return chainid2genomeid;
}

//------------------------------------------ Python glue ------------------------------------------

PyObject *get_column(PyObject *dict, const char *key, int dtype)
{
    PyObject *item = PyDict_GetItemString(dict, key);
    if (!item)
        verror("chain is missing column '%s'", key);
    PyObject *arr = PyArray_FROM_OTF(item, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        verror("chain column '%s' has a wrong type", key);
    if (PyArray_NDIM((PyArrayObject *)arr) != 1) {
        Py_DECREF(arr);
        verror("chain column '%s' must be 1-D", key);
    }
    return arr;
}

void get_strings(PyObject *obj, const char *what, std::vector<std::string> &out)
{
    if (!obj)
        verror("%s is missing", what);
    PMPY seq(PySequence_Fast(obj, what), true);
    if (!seq)
        verror("%s must be a sequence", what);
    Py_ssize_t n = PySequence_Fast_GET_SIZE(*seq);
    out.clear();
    out.reserve(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(*seq, i);
        Py_ssize_t len = 0;
        const char *str = PyUnicode_Check(item) ? PyUnicode_AsUTF8AndSize(item, &len) : nullptr;
        if (!str)
            verror("%s must hold strings", what);
        out.emplace_back(str, (size_t)len);
    }
}

}  // namespace

// pm_liftover_track_2d(src_track_dir, chain, track_dir, tgt_chroms, tgt_sizes, tgt_overlap_policy,
//                      multi_target_agg, na_rm, min_n, nth_index) -> "points" | "rects" | ""
//
// chain: dict of chain columns (gintervals_load_chain). track_dir: the (staging) directory
// of the new track, which must exist; the pair files are written there. tgt_chroms,
// tgt_sizes: the chromosomes of the current database in chrom id order. min_n < 0: none.
// Returns the type of the source, or "" if no chromosome pair of it was read.
PyObject *pm_liftover_track_2d(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *src_track_dir_str, *track_dir_str, *tgt_policy_str, *agg_str;
        PyObject *chain_dict, *tgt_chroms_obj, *tgt_sizes_obj;
        int na_rm;
        long long min_n, nth_index;

        PyObject *src_db_chroms_obj = nullptr;
        if (!PyArg_ParseTuple(args, "sOsOOsspLL|O", &src_track_dir_str, &chain_dict, &track_dir_str, &tgt_chroms_obj,
                              &tgt_sizes_obj, &tgt_policy_str, &agg_str, &na_rm, &min_n, &nth_index, &src_db_chroms_obj))
            return nullptr;
        if (!PyDict_Check(chain_dict))
            verror("chain must be a dict");

        const std::string src_track_dir(src_track_dir_str);
        const std::string dirname(track_dir_str);

        AggregationConfig agg_cfg;
        agg_cfg.type = parse_agg_type_str(agg_str);
        agg_cfg.na_rm = na_rm != 0;
        agg_cfg.min_n = min_n < 0 ? -1 : (int64_t)min_n;
        agg_cfg.nth_index = agg_cfg.type == AggType::NTH ? (int64_t)nth_index : -1;
        if (agg_cfg.type == AggType::NTH && agg_cfg.nth_index <= 0)
            verror("params must be a positive integer for 'nth' aggregation");

        // the target genome
        std::vector<std::string> tgt_chroms;
        get_strings(tgt_chroms_obj, "tgt_chroms", tgt_chroms);
        std::vector<int64_t> tgt_sizes;
        {
            PMPY seq(PySequence_Fast(tgt_sizes_obj, "tgt_sizes"), true);
            if (!seq || PySequence_Fast_GET_SIZE(*seq) != (Py_ssize_t)tgt_chroms.size())
                verror("tgt_sizes must hold a size per chromosome");
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*seq); ++i) {
                long long size = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(*seq, i));
                if (size == -1 && PyErr_Occurred())
                    return nullptr;
                tgt_sizes.push_back(size);
            }
        }
        std::unordered_map<std::string, int> tgt_chrom2id;
        for (size_t i = 0; i < tgt_chroms.size(); ++i)
            tgt_chrom2id.emplace(tgt_chroms[i], (int)i);

        // the chain
        std::vector<std::string> chrom, chromsrc;
        get_strings(PyDict_GetItemString(chain_dict, "chrom"), "chain column 'chrom'", chrom);
        get_strings(PyDict_GetItemString(chain_dict, "chromsrc"), "chain column 'chromsrc'", chromsrc);
        PMPY start(get_column(chain_dict, "start", NPY_INT64), true);
        PMPY end(get_column(chain_dict, "end", NPY_INT64), true);
        PMPY strand(get_column(chain_dict, "strand", NPY_INT64), true);
        PMPY startsrc(get_column(chain_dict, "startsrc", NPY_INT64), true);
        PMPY endsrc(get_column(chain_dict, "endsrc", NPY_INT64), true);
        PMPY chain_id(get_column(chain_dict, "chain_id", NPY_INT64), true);
        PMPY score(get_column(chain_dict, "score", NPY_DOUBLE), true);
        const size_t num_chain = chrom.size();
        for (const PMPY *col : { &start, &end, &strand, &startsrc, &endsrc, &chain_id, &score }) {
            if ((size_t)PyArray_DIM((PyArrayObject *)**col, 0) != num_chain)
                verror("chain columns have different lengths");
        }
        if (chromsrc.size() != num_chain)
            verror("chain columns have different lengths");

        auto int64_col = [](const PMPY &col, size_t i) { return ((const int64_t *)PyArray_DATA((PyArrayObject *)*col))[i]; };
        std::vector<ChainInterval> chain_intervs(num_chain);
        for (size_t i = 0; i < num_chain; ++i) {
            ChainInterval &c = chain_intervs[i];
            auto ichrom = tgt_chrom2id.find(chrom[i]);
            if (ichrom == tgt_chrom2id.end())
                verror("Chain target chromosome %s is not in the database", chrom[i].c_str());
            c.chromid = ichrom->second;
            c.start = int64_col(start, i);
            c.end = int64_col(end, i);
            c.strand = int64_col(strand, i);
            c.start_src = int64_col(startsrc, i);
            c.end_src = int64_col(endsrc, i);
            c.chain_id = int64_col(chain_id, i);
            c.score = ((const double *)PyArray_DATA((PyArrayObject *)*score))[i];
            c.chromid_src = -1;
        }

        // The source chromosomes are numbered in the order they first appear in the chain,
        // which R's gintervals.load_chain returns sorted by target. That numbering is the
        // order in which the source pairs are read, and so the order first / last / nth follow.
        std::vector<size_t> tgt_order(num_chain);
        for (size_t i = 0; i < num_chain; ++i)
            tgt_order[i] = i;
        std::stable_sort(tgt_order.begin(), tgt_order.end(), [&](size_t a, size_t b) {
            const ChainInterval &x = chain_intervs[a], &y = chain_intervs[b];
            if (x.chromid != y.chromid)
                return x.chromid < y.chromid;
            if (x.start != y.start)
                return x.start < y.start;
            return x.end < y.end;
        });
        std::vector<std::string> src_id2chrom;
        {
            std::unordered_map<std::string, int> src_chrom2id;
            for (size_t i : tgt_order) {
                auto ichrom = src_chrom2id.find(chromsrc[i]);
                if (ichrom == src_chrom2id.end()) {
                    ichrom = src_chrom2id.emplace(chromsrc[i], (int)src_id2chrom.size()).first;
                    src_id2chrom.push_back(chromsrc[i]);
                }
                chain_intervs[i].chromid_src = ichrom->second;
            }
        }

        ChainIntervals chain;
        chain.init(std::move(chain_intervs), (int)src_id2chrom.size(), tgt_policy_str);

        // the source track
        struct stat st;
        if (stat(src_track_dir.c_str(), &st) || !S_ISDIR(st.st_mode))
            throw std::runtime_error("Source track 2D directory does not exist: " + src_track_dir);

        const std::string idx_path = src_track_dir + "/track.idx";
        const bool src_indexed = !stat(idx_path.c_str(), &st);
        TrackIndex2D src_index;
        std::vector<int> src_chainid2genomeid;
        int32_t src_signature = 0;

        if (src_indexed) {
            src_index.load(idx_path);
            src_signature = src_index.get_track_type() == MishaTrack2DType::POINTS ? POINTS_SIGNATURE : RECTS_SIGNATURE;

            // an indexed track is keyed by the chrom ids of the database that holds it
            std::string db_root;
            if (!source_db_root_from_track_dir(src_track_dir, db_root)) {
                char buf[1024];
                snprintf(buf, sizeof(buf),
                    "Indexed source track %s is not in a database (a directory with chrom_sizes.txt "
                    "whose \"tracks\" directory contains the track); an indexed track is read by the "
                    "chromosome ids of its database", src_track_dir.c_str());
                throw std::invalid_argument(buf);
            }
            // the chromosomes the caller read for its database (R's .gdb.chrom_names_at)
            std::vector<std::string> src_chroms;
            if (src_db_chroms_obj && src_db_chroms_obj != Py_None)
                get_strings(src_db_chroms_obj, "src_db_chroms", src_chroms);
            else
                load_source_chrom_names(db_root, src_chroms);
            src_chainid2genomeid = map_src_chainid2genomeid(src_id2chrom, src_chroms);
        } else
            src_signature = per_pair_track_signature(src_track_dir);

        // The chain's source names, and as their aliases the names with the "chr" prefix
        // toggled: a per-chromosome source track may name its files either way (as earlier
        // pymisha did); its files are the ones every reader finds (GenomeTrack::ChromFiles)
        GenomeChromKey src_chromkey;
        chain_src_chromkey(src_id2chrom, src_chromkey);
        std::unique_ptr<GenomeTrack::ChromFiles> src_files;
        if (!src_indexed) {
            src_files.reset(new GenomeTrack::ChromFiles(src_chromkey, src_track_dir));
            src_files->chrom_pairs();  // two files of a pair are an error here
        }

        if (!src_signature)
            return PyUnicode_FromString("");
        const bool is_points = src_signature == POINTS_SIGNATURE;

        // pass 1: map the source objects into a buffer file per target chromosome pair
        std::map<std::pair<int, int>, std::unique_ptr<BufferedFile>> buffers;
        auto buffer = [&](int chromid1, int chromid2) -> BufferedFile & {
            std::unique_ptr<BufferedFile> &bfile = buffers[std::make_pair(chromid1, chromid2)];
            if (!bfile) {
                bfile.reset(new BufferedFile());
                open_chk(*bfile, dirname + "/_" + tgt_chroms[chromid1] + "-" + tgt_chroms[chromid2], "w+");
            }
            return *bfile;
        };

        BufferedFile src_dat;
        if (src_indexed)
            open_chk(src_dat, src_track_dir + "/track.dat", "rb");

        std::vector<TgtInterval> tgt_intervals[2];
        bool read_any = false;

        for (int chromid1 = 0; chromid1 < (int)src_id2chrom.size(); ++chromid1) {
            for (int chromid2 = 0; chromid2 < (int)src_id2chrom.size(); ++chromid2) {
                BufferedFile pair_file;
                BufferedFile *src_file;
                int64_t base_offset = 0;

                if (src_indexed) {
                    int read_chromid1 = src_chainid2genomeid[chromid1];
                    int read_chromid2 = src_chainid2genomeid[chromid2];
                    if (read_chromid1 < 0 || read_chromid2 < 0)   // not in the source genome
                        continue;
                    const Track2DPairEntry *entry = src_index.get_entry((uint32_t)read_chromid1, (uint32_t)read_chromid2);
                    if (!entry || !entry->length)
                        continue;
                    if (src_dat.seek((int64_t)entry->offset, SEEK_SET))
                        TGLError("Failed to seek to offset %llu in %s", (unsigned long long)entry->offset, src_dat.file_name().c_str());
                    base_offset = (int64_t)entry->offset;
                    src_file = &src_dat;
                } else {
                    if (!src_files->exists(chromid1, chromid2))
                        continue;
                    std::string filename = src_files->path(chromid1, chromid2);
                    open_chk(pair_file, filename, "rb");
                    src_file = &pair_file;
                }

                int32_t signature;
                if (src_file->read(&signature, sizeof(signature)) != sizeof(signature))
                    TGLError("Invalid format of a 2D track file %s", src_file->file_name().c_str());
                if (signature != src_signature)
                    TGLError("Source track %s mixes RECTS and POINTS chromosome pairs", src_track_dir.c_str());
                read_any = true;

                if (is_points) {
                    PointsQuadTreeCached qtree(TRACK_CHUNK_SIZE, SRC_NUM_CHUNKS);
                    qtree.set_base_offset(base_offset);
                    qtree.unserialize(*src_file);
                    if (qtree.empty())
                        continue;

                    PointsQuadTreeCached::Iterator iqtree(&qtree);
                    for (iqtree.begin(); !iqtree.is_end(); iqtree.next()) {
                        chain.map_interval(chromid1, iqtree->x, iqtree->x + 1, tgt_intervals[0]);
                        chain.map_interval(chromid2, iqtree->y, iqtree->y + 1, tgt_intervals[1]);
                        for (const TgtInterval &interv1 : tgt_intervals[0]) {
                            for (const TgtInterval &interv2 : tgt_intervals[1])
                                write_point(buffer(interv1.chromid, interv2.chromid), interv1.start, interv2.start, iqtree->v);
                        }
                        check_interrupt();
                    }
                } else {
                    RectsQuadTreeCached qtree(TRACK_CHUNK_SIZE, SRC_NUM_CHUNKS);
                    qtree.set_base_offset(base_offset);
                    qtree.unserialize(*src_file);
                    if (qtree.empty())
                        continue;

                    RectsQuadTreeCached::Iterator iqtree(&qtree);
                    for (iqtree.begin(); !iqtree.is_end(); iqtree.next()) {
                        chain.map_interval(chromid1, iqtree->x1, iqtree->x2, tgt_intervals[0]);
                        chain.map_interval(chromid2, iqtree->y1, iqtree->y2, tgt_intervals[1]);
                        for (const TgtInterval &interv1 : tgt_intervals[0]) {
                            for (const TgtInterval &interv2 : tgt_intervals[1])
                                write_rect(buffer(interv1.chromid, interv2.chromid), interv1, interv2, iqtree->v);
                        }
                        check_interrupt();
                    }
                }
            }
        }
        src_dat.close();

        // Close every pair buffer before the pairs are written, and reopen each when its
        // pair is written: a pair split into subtrees opens a file per subtree, and with all
        // the buffers still open hundreds of target pairs could exceed the open-file limit.
        std::vector<std::pair<std::pair<int, int>, std::string>> pairs;
        for (auto &kv : buffers) {
            flush_chk(*kv.second, "intervals");
            pairs.emplace_back(kv.first, kv.second->file_name());
            if (kv.second->close())
                TGLError("Failed to write intervals to file %s: %s", pairs.back().second.c_str(), strerror(errno));
        }
        buffers.clear();

        // pass 2: write the pairs one by one; each buffer is removed once it is read
        const uint64_t max_data_size = pymisha.max_data_size();
        for (const auto &pair : pairs) {
            int chromid1 = pair.first.first;
            int chromid2 = pair.first.second;
            std::string filename = dirname + "/" + tgt_chroms[chromid1] + "-" + tgt_chroms[chromid2];

            if (is_points)
                write_2d_points(pair.second, filename, dirname, tgt_chroms[chromid1], tgt_chroms[chromid2],
                                tgt_sizes[chromid1], tgt_sizes[chromid2], max_data_size, agg_cfg);
            else
                write_2d_rects(pair.second, filename, tgt_sizes[chromid1], tgt_sizes[chromid2], agg_cfg);
        }

        return PyUnicode_FromString(read_any ? (is_points ? "points" : "rects") : "");

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}
