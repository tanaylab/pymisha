// pm_read_source_track_1d: C++ port of liftover._read_source_track (1D path).
//
// Reads a source-track directory and returns a tuple (track_type, df_dict)
// where df_dict is a numpy-array dict with columns chrom, start, end, value.
// Handles both per-chrom binary files (dense sig>0 or sparse sig=-1) and the
// indexed track.idx+track.dat pair. Source chrom names are NOT normalized to
// the target chromkey - chain matching happens later (P3.B).
//
// The aggregation + lift inner loop of gtrack_liftover is NOT done here -
// that stays in Python until G1.P3.B + P3.C.

#include "pymisha.h"
#include "PMSourceTrack.h"
#include "CRC64.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

// Decide whether a directory entry name is a candidate per-chrom track file.
// Skip leading-dot files (.attributes etc.) and the indexed pair members.
bool is_per_chrom_candidate(const char *name) {
    if (!name || !*name) return false;
    if (name[0] == '.') return false;
    if (!strcmp(name, "track.idx")) return false;
    if (!strcmp(name, "track.dat")) return false;
    return true;
}

// Read an entire file into a byte vector.
bool slurp_file(const std::string &path, std::vector<char> &out)
{
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0) { fclose(fp); return false; }
    size_t nread = 0;
    try {
        out.resize((size_t)sz);
        nread = sz > 0 ? fread(out.data(), 1, (size_t)sz, fp) : 0;
    } catch (...) {
        fclose(fp);
        throw;
    }
    fclose(fp);
    return nread == (size_t)sz;
}

// Read sorted candidate file names from a directory.
bool list_dir_sorted(const std::string &dir, std::vector<std::string> &out)
{
    DIR *d = opendir(dir.c_str());
    if (!d) return false;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        out.emplace_back(e->d_name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return true;
}

// ---- Indexed format (track.idx + track.dat) helpers ----

constexpr int64_t TRACK_IDX_MAGIC_LEN = 8;
const char TRACK_IDX_MAGIC[TRACK_IDX_MAGIC_LEN + 1] = "MISHATDX";
constexpr uint32_t TRACK_IDX_VERSION = 1;
constexpr uint64_t TRACK_IDX_FLAG_LITTLE_ENDIAN = 0x01ULL;
// MishaTrackType enum values (matches TrackIndex.h: DENSE=0, SPARSE=1)
constexpr uint32_t TRACK_TYPE_DENSE_RAW  = 0;
constexpr uint32_t TRACK_TYPE_SPARSE_RAW = 1;

struct TrackIdxEntry {
    uint32_t chrom_id;
    uint64_t offset;
    uint64_t length;
    uint32_t reserved;
};

// Compute CRC64 over the entries: for each entry feed (u32 chrom_id LE,
// u64 offset LE, u64 length LE) -- matches _compute_track_idx_checksum in
// liftover.py and the PMTrackIndexedFormat.cpp writer.
uint64_t compute_idx_checksum(const std::vector<TrackIdxEntry> &entries)
{
    misha::CRC64 crc64;
    uint64_t crc = crc64.init_incremental();
    for (const auto &e : entries) {
        uint32_t cid = e.chrom_id;
        uint64_t off = e.offset;
        uint64_t len = e.length;
        crc = crc64.compute_incremental(crc, (const unsigned char*)&cid, sizeof(cid));
        crc = crc64.compute_incremental(crc, (const unsigned char*)&off, sizeof(off));
        crc = crc64.compute_incremental(crc, (const unsigned char*)&len, sizeof(len));
    }
    return crc64.finalize_incremental(crc);
}

// Walk up from src_track_dir until a path component named "tracks" is found;
// return its parent as db_root. Returns false if not found.
bool source_db_root_from_track_dir(const std::string &src_track_dir, std::string &db_root_out)
{
    std::string p = src_track_dir;
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    while (true) {
        size_t slash = p.find_last_of('/');
        if (slash == std::string::npos) return false;
        std::string parent = p.substr(0, slash);
        std::string base = p.substr(slash + 1);
        if (base == "tracks") {
            db_root_out = parent;
            return true;
        }
        p = parent;
        if (p.empty()) return false;
    }
}

// Load chrom_id -> chrom_name list from <db_root>/chrom_sizes.txt.
// Names are indexed by line order (0-based). Throws std::invalid_argument on
// failure (missing file, malformed line).
void load_source_chrom_names(const std::string &db_root, std::vector<std::string> &names_out)
{
    std::string path = db_root + "/chrom_sizes.txt";
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp) {
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "Cannot resolve chromosome IDs for indexed source track: missing %s",
            path.c_str());
        throw std::invalid_argument(buf);
    }
    char *line = nullptr;
    size_t cap = 0;
    ssize_t nread = 0;
    while ((nread = getline(&line, &cap, fp)) != -1) {
        // Strip trailing whitespace including newlines.
        while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r' ||
                              line[nread - 1] == ' '  || line[nread - 1] == '\t')) {
            line[--nread] = '\0';
        }
        if (nread == 0) continue;
        // Skip leading whitespace.
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p) continue;
        // Find end of first token (chrom name).
        char *q = p;
        while (*q && *q != ' ' && *q != '\t') ++q;
        bool has_size = (*q != '\0');
        *q = '\0';
        if (!has_size) {
            std::string msg;
            char buf[1024];
            snprintf(buf, sizeof(buf), "Invalid line in %s: %s", path.c_str(), p);
            free(line);
            fclose(fp);
            throw std::invalid_argument(buf);
        }
        names_out.emplace_back(p);
    }
    if (line) free(line);
    fclose(fp);
}

// ---- End indexed format helpers ----

// Decode a dense payload: int32 bin_size + float32 array.
// Append valid (non-NaN, non-Inf) bins to the output vectors. chrom_name is
// the per-chrom file name (per-chrom case) or chrom_sizes.txt name (indexed).
// out_bin_size is set to the bin size extracted from the payload header.
void decode_dense_payload(
    const std::string &chrom_name,
    const std::vector<char> &payload,
    std::vector<std::string> &chroms_out,
    std::vector<int64_t> &starts_out,
    std::vector<int64_t> &ends_out,
    std::vector<double> &values_out,
    const std::string &source_label,
    int32_t &out_bin_size)
{
    if (payload.size() < 4) {
        out_bin_size = 0;
        return;
    }
    if ((payload.size() - 4) % 4 != 0) {
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "Corrupt dense track payload for %s in %s",
            chrom_name.c_str(), source_label.c_str());
        throw std::invalid_argument(buf);
    }
    int32_t bin_size;
    memcpy(&bin_size, payload.data(), sizeof(int32_t));
    if (bin_size <= 0) {
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "Invalid dense bin size for %s in %s: %d",
            chrom_name.c_str(), source_label.c_str(), bin_size);
        throw std::invalid_argument(buf);
    }

    out_bin_size = bin_size;

    const size_t n_bins = (payload.size() - 4) / 4;
    const float *floats = reinterpret_cast<const float *>(payload.data() + 4);
    for (size_t i = 0; i < n_bins; ++i) {
        float v = floats[i];
        if (std::isnan(v) || std::isinf(v)) continue;
        chroms_out.push_back(chrom_name);
        int64_t start = (int64_t)i * (int64_t)bin_size;
        starts_out.push_back(start);
        ends_out.push_back(start + (int64_t)bin_size);
        values_out.push_back(static_cast<double>(v));
    }
}

// Decode a sparse payload. Sig=-1 was already validated by the caller.
// The body alternates either 64-bit (i8 start, i8 end, f4 value) or 32-bit
// (i4, i4, f4). Python disambiguates by: try both decode shapes; if the
// length divides both record sizes AND both pass validity (start>=0,
// end>=start), prefer 64-bit. Otherwise use the layout that validates.
void decode_sparse_payload(
    const std::string &chrom_name,
    const std::vector<char> &payload,
    std::vector<std::string> &chroms_out,
    std::vector<int64_t> &starts_out,
    std::vector<int64_t> &ends_out,
    std::vector<double> &values_out,
    const std::string &source_label)
{
    if (payload.size() < 4) return;  // No signature, treat as empty.
    if (payload.size() == 4) return;  // Sig only, no records.

    const char *body = payload.data() + 4;
    const size_t body_len = payload.size() - 4;

    const size_t SIZE_64 = 8 + 8 + 4;  // 20 bytes
    const size_t SIZE_32 = 4 + 4 + 4;  // 12 bytes

    bool can64 = (body_len % SIZE_64) == 0;
    bool can32 = (body_len % SIZE_32) == 0;
    if (!can64 && !can32) {
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "Corrupt sparse track payload length for %s in %s: %zu bytes",
            chrom_name.c_str(), source_label.c_str(), body_len);
        throw std::invalid_argument(buf);
    }

    // Try 64-bit first if it can validate.
    auto try_decode_64 = [&](bool &valid_out, size_t &n_recs_out) {
        size_t n = body_len / SIZE_64;
        valid_out = true;
        for (size_t i = 0; i < n; ++i) {
            int64_t s, e;
            memcpy(&s, body + i * SIZE_64,     sizeof(int64_t));
            memcpy(&e, body + i * SIZE_64 + 8, sizeof(int64_t));
            if (s < 0 || e < s) { valid_out = false; break; }
        }
        n_recs_out = n;
    };
    auto try_decode_32 = [&](bool &valid_out, size_t &n_recs_out) {
        size_t n = body_len / SIZE_32;
        valid_out = true;
        for (size_t i = 0; i < n; ++i) {
            int32_t s, e;
            memcpy(&s, body + i * SIZE_32,     sizeof(int32_t));
            memcpy(&e, body + i * SIZE_32 + 4, sizeof(int32_t));
            if (s < 0 || e < s) { valid_out = false; break; }
        }
        n_recs_out = n;
    };

    bool valid64 = false, valid32 = false;
    size_t n64 = 0, n32 = 0;
    if (can64) try_decode_64(valid64, n64);
    if (can32) try_decode_32(valid32, n32);

    bool use_64 = false;
    if (can64 && can32) {
        if (valid64 && !valid32)        use_64 = true;
        else if (valid32 && !valid64)   use_64 = false;
        else if (valid64 && valid32)    use_64 = true;  // prefer 64-bit
        else {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Corrupt sparse track payload records for %s in %s",
                chrom_name.c_str(), source_label.c_str());
            throw std::invalid_argument(buf);
        }
    } else if (can64) {
        if (!valid64) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Invalid sparse 64-bit records for %s in %s",
                chrom_name.c_str(), source_label.c_str());
            throw std::invalid_argument(buf);
        }
        use_64 = true;
    } else {
        if (!valid32) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Invalid sparse 32-bit records for %s in %s",
                chrom_name.c_str(), source_label.c_str());
            throw std::invalid_argument(buf);
        }
        use_64 = false;
    }

    if (use_64) {
        for (size_t i = 0; i < n64; ++i) {
            int64_t s, e;
            float v;
            memcpy(&s, body + i * SIZE_64,      sizeof(int64_t));
            memcpy(&e, body + i * SIZE_64 + 8,  sizeof(int64_t));
            memcpy(&v, body + i * SIZE_64 + 16, sizeof(float));
            if (std::isnan(v) || std::isinf(v)) continue;
            chroms_out.push_back(chrom_name);
            starts_out.push_back(s);
            ends_out.push_back(e);
            values_out.push_back(static_cast<double>(v));
        }
    } else {
        for (size_t i = 0; i < n32; ++i) {
            int32_t s, e;
            float v;
            memcpy(&s, body + i * SIZE_32,     sizeof(int32_t));
            memcpy(&e, body + i * SIZE_32 + 4, sizeof(int32_t));
            memcpy(&v, body + i * SIZE_32 + 8, sizeof(float));
            if (std::isnan(v) || std::isinf(v)) continue;
            chroms_out.push_back(chrom_name);
            starts_out.push_back(s);
            ends_out.push_back(e);
            values_out.push_back(static_cast<double>(v));
        }
    }
}

} // namespace

// ---- Public C++ helper (declared in PMSourceTrack.h) ----

void read_source_track_1d_cpp(
    const std::string &src_track_dir,
    std::string &out_type,
    std::int64_t &out_bin_size,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value)
{
    std::vector<std::string> entries;
    if (!list_dir_sorted(src_track_dir, entries)) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "Failed to list directory: %s", src_track_dir.c_str());
        throw std::runtime_error(buf);
    }

    bool has_indexed = false;
    {
        bool has_idx = false, has_dat = false;
        for (const auto &n : entries) {
            if (n == "track.idx") has_idx = true;
            else if (n == "track.dat") has_dat = true;
        }
        has_indexed = has_idx && has_dat;
    }
    std::vector<std::string> per_chrom_files;
    for (const auto &n : entries) {
        if (!is_per_chrom_candidate(n.c_str())) continue;
        // Only regular files are per-chrom data files; skip subdirs like 'vars'.
        std::string fpath = src_track_dir + "/" + n;
        struct stat fst;
        if (stat(fpath.c_str(), &fst) != 0 || !S_ISREG(fst.st_mode)) continue;
        per_chrom_files.push_back(n);
    }

    std::string track_type;  // "" | "dense" | "sparse"
    int32_t prev_bin_size = -1;
    std::string prev_fname;

    // If only the indexed pair is present (no per-chrom files), use the
    // indexed reader. Otherwise iterate per-chrom files.
    if (per_chrom_files.empty() && has_indexed) {
        std::string idx_path = src_track_dir + "/track.idx";
        std::string dat_path = src_track_dir + "/track.dat";

        std::vector<char> idx_buf;
        if (!slurp_file(idx_path, idx_buf)) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Indexed source track is missing track.idx/track.dat in %s",
                src_track_dir.c_str());
            throw std::invalid_argument(buf);
        }
        if (idx_buf.size() < 36) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Truncated track.idx header in %s", idx_path.c_str());
            throw std::invalid_argument(buf);
        }
        if (memcmp(idx_buf.data(), TRACK_IDX_MAGIC, TRACK_IDX_MAGIC_LEN) != 0) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Invalid track index header in %s", idx_path.c_str());
            throw std::invalid_argument(buf);
        }

        uint32_t version = 0, track_type_raw = 0, num_contigs = 0;
        uint64_t flags = 0, stored_checksum = 0;
        memcpy(&version,         idx_buf.data() + 8,  4);
        memcpy(&track_type_raw,  idx_buf.data() + 12, 4);
        memcpy(&num_contigs,     idx_buf.data() + 16, 4);
        memcpy(&flags,           idx_buf.data() + 20, 8);
        memcpy(&stored_checksum, idx_buf.data() + 28, 8);

        if (version != TRACK_IDX_VERSION) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Unsupported track index version %u in %s",
                (unsigned)version, idx_path.c_str());
            throw std::invalid_argument(buf);
        }
        if ((flags & TRACK_IDX_FLAG_LITTLE_ENDIAN) == 0) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Unsupported track index endianness in %s", idx_path.c_str());
            throw std::invalid_argument(buf);
        }
        if (track_type_raw != TRACK_TYPE_DENSE_RAW && track_type_raw != TRACK_TYPE_SPARSE_RAW) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Unsupported indexed source track type %u in %s",
                (unsigned)track_type_raw, src_track_dir.c_str());
            throw std::invalid_argument(buf);
        }

        const size_t entries_offset = 36;
        const size_t entries_size = (size_t)num_contigs * 24;
        if (idx_buf.size() < entries_offset + entries_size) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Truncated track index entries in %s", idx_path.c_str());
            throw std::invalid_argument(buf);
        }

        std::vector<TrackIdxEntry> idx_entries;
        idx_entries.reserve(num_contigs);
        for (uint32_t i = 0; i < num_contigs; ++i) {
            const char *p = idx_buf.data() + entries_offset + (size_t)i * 24;
            TrackIdxEntry e;
            memcpy(&e.chrom_id, p,      4);
            memcpy(&e.offset,   p + 4,  8);
            memcpy(&e.length,   p + 12, 8);
            memcpy(&e.reserved, p + 20, 4);
            idx_entries.push_back(e);
        }

        uint64_t computed = compute_idx_checksum(idx_entries);
        if (computed != stored_checksum) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "track.idx checksum mismatch in %s (expected %016llx, got %016llx)",
                idx_path.c_str(),
                (unsigned long long)stored_checksum,
                (unsigned long long)computed);
            throw std::invalid_argument(buf);
        }

        std::string db_root;
        if (!source_db_root_from_track_dir(src_track_dir, db_root)) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Indexed source track path must be located under a database tracks directory "
                "(got: %s)", src_track_dir.c_str());
            throw std::invalid_argument(buf);
        }
        std::vector<std::string> chrom_names;
        load_source_chrom_names(db_root, chrom_names);  // throws on error

        if (track_type_raw == TRACK_TYPE_DENSE_RAW) track_type = "dense";
        else                                          track_type = "sparse";

        FILE *dat_fp = fopen(dat_path.c_str(), "rb");
        if (!dat_fp) {
            char buf[1024];
            snprintf(buf, sizeof(buf),
                "Indexed source track is missing track.idx/track.dat in %s",
                src_track_dir.c_str());
            throw std::invalid_argument(buf);
        }

        // For indexed-dense, extract bin_size from the first non-empty payload.
        // All indexed chroms share the same bin_size (file-level property).
        int32_t indexed_bin_size = 0;

        try {
            for (const auto &e : idx_entries) {
                if (e.length == 0) continue;
                if (e.chrom_id >= chrom_names.size()) {
                    fclose(dat_fp);
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "Indexed source track has chrom_id=%u not present in source chrom_sizes.txt",
                        (unsigned)e.chrom_id);
                    throw std::invalid_argument(buf);
                }
                if (fseek(dat_fp, (long)e.offset, SEEK_SET) != 0) {
                    fclose(dat_fp);
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "Failed to seek in %s", dat_path.c_str());
                    throw std::runtime_error(buf);
                }
                std::vector<char> payload((size_t)e.length);
                size_t got = fread(payload.data(), 1, payload.size(), dat_fp);
                if (got != payload.size()) {
                    fclose(dat_fp);
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "Failed to read %llu bytes for chrom_id=%u from %s",
                        (unsigned long long)e.length,
                        (unsigned)e.chrom_id, dat_path.c_str());
                    throw std::invalid_argument(buf);
                }
                if (track_type_raw == TRACK_TYPE_DENSE_RAW) {
                    int32_t this_bin_size = 0;
                    decode_dense_payload(chrom_names[e.chrom_id], payload,
                                        out_chrom, out_start, out_end, out_value,
                                        "indexed source track", this_bin_size);
                    // Capture bin_size from first non-empty payload (all share the same).
                    if (this_bin_size > 0 && indexed_bin_size == 0) {
                        indexed_bin_size = this_bin_size;
                    }
                } else {
                    decode_sparse_payload(chrom_names[e.chrom_id], payload,
                                         out_chrom, out_start, out_end, out_value,
                                         "indexed source track");
                }
            }
        } catch (...) {
            fclose(dat_fp);
            throw;
        }
        fclose(dat_fp);

        prev_bin_size = indexed_bin_size;
    } else {
        for (const auto &fname : per_chrom_files) {
            std::string fpath = src_track_dir + "/" + fname;
            struct stat ist;
            if (stat(fpath.c_str(), &ist) != 0 || !S_ISREG(ist.st_mode)) continue;

            std::vector<char> payload;
            if (!slurp_file(fpath, payload)) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "Failed to read file: %s", fpath.c_str());
                throw std::runtime_error(buf);
            }
            if (payload.size() < 4) continue;
            int32_t sig;
            memcpy(&sig, payload.data(), sizeof(int32_t));

            if (sig > 0) {
                // Dense layout: sig is the bin_size; payload = int32 bin_size + float32[].
                if (track_type.empty()) track_type = "dense";
                else if (track_type != "dense") {
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "Mixed dense/sparse source files in %s", src_track_dir.c_str());
                    throw std::invalid_argument(buf);
                }
                int32_t this_bin_size = 0;
                decode_dense_payload(fname, payload, out_chrom, out_start, out_end, out_value,
                                     "per-chrom source track", this_bin_size);
                // R-parity: all dense per-chrom files must have the same bin_size.
                if (this_bin_size > 0) {
                    if (prev_bin_size > 0 && this_bin_size != prev_bin_size) {
                        char buf[1024];
                        snprintf(buf, sizeof(buf),
                            "Binsize of track file %s differs from the binsize of track file %s (%d vs. %d)",
                            fname.c_str(), prev_fname.c_str(), this_bin_size, prev_bin_size);
                        throw std::invalid_argument(buf);
                    }
                    prev_bin_size = this_bin_size;
                    prev_fname = fname;
                }
            } else if (sig == -1) {
                if (track_type.empty()) track_type = "sparse";
                else if (track_type != "sparse") {
                    char buf[1024];
                    snprintf(buf, sizeof(buf),
                        "Mixed dense/sparse source files in %s", src_track_dir.c_str());
                    throw std::invalid_argument(buf);
                }
                decode_sparse_payload(fname, payload, out_chrom, out_start, out_end, out_value,
                                      "per-chrom source track");
            }
            // Other signatures: silently skip (matches Python which only handles
            // sig > 0 and sig == -1).
        }
    }

    if (track_type.empty()) track_type = "sparse";
    out_type = track_type;
    out_bin_size = (prev_bin_size > 0) ? (int64_t)prev_bin_size : 0;
}

/*
 * pm_read_source_track_1d(src_track_dir) -> tuple(str track_type, dict df) | None
 *
 * src_track_dir : str - filesystem path to a source-track directory.
 *
 * Returns: 2-tuple (track_type, df_dict).
 *   track_type : "dense" | "sparse".
 *   df_dict    : numpy-array dict with keys "chrom" (NPY_OBJECT[str]),
 *                "start"/"end" (NPY_INT64), "value" (NPY_FLOAT64). All arrays
 *                have the same length N. N==0 is valid (empty track).
 *
 * Raises:
 *   ValueError - src_track_dir does not exist, payload is corrupt, dense+sparse
 *                mixed in per-chrom files, indexed-format errors (bad magic,
 *                bad version, checksum mismatch, ...).
 *   RuntimeError - I/O failure.
 */
PyObject *pm_read_source_track_1d(PyObject *self, PyObject *args)
{
    const char *src_track_dir;
    if (!PyArg_ParseTuple(args, "s", &src_track_dir)) {
        return nullptr;
    }

    struct stat st;
    if (stat(src_track_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        PyErr_Format(PyExc_ValueError,
            "Source track directory does not exist: %s", src_track_dir);
        return nullptr;
    }

    std::string out_type;
    std::int64_t out_bin_size = 0;
    std::vector<std::string> chroms;
    std::vector<int64_t> starts, ends;
    std::vector<double> values;

    try {
        read_source_track_1d_cpp(
            src_track_dir, out_type, out_bin_size, chroms, starts, ends, values);
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    } catch (const std::invalid_argument &e) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::runtime_error &e) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (const std::exception &e) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    const std::string &track_type = out_type;

    npy_intp n = (npy_intp)chroms.size();
    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_value(PyArray_SimpleNew(1, &n, NPY_FLOAT64), true);
    if (!py_chrom || !py_start || !py_end || !py_value) return_err();

    PyObject **chrom_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    for (npy_intp i = 0; i < n; ++i) chrom_out[i] = nullptr;
    int64_t *start_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    double  *value_out = (double *) PyArray_DATA((PyArrayObject *)*py_value);

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(chroms[i].data(), chroms[i].size());
        if (!s) {
            for (npy_intp j = i; j < n; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
            }
            return_err();
        }
        chrom_out[i] = s;
        start_out[i] = starts[i];
        end_out[i]   = ends[i];
        value_out[i] = values[i];
    }

    PMPY df(PyDict_New(), true);
    if (!df) return_err();
    PyDict_SetItemString(df, "chrom", py_chrom);
    PyDict_SetItemString(df, "start", py_start);
    PyDict_SetItemString(df, "end",   py_end);
    PyDict_SetItemString(df, "value", py_value);

    PMPY result(PyTuple_New(2), true);
    if (!result) return_err();
    PyObject *tt = PyUnicode_FromString(track_type.c_str());
    if (!tt) return_err();
    PyTuple_SET_ITEM(*result, 0, tt);
    Py_INCREF(*df);
    PyTuple_SET_ITEM(*result, 1, *df);

    return_py(result);
}
