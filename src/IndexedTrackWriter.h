#ifndef PYMISHA_INDEXED_TRACK_WRITER_H
#define PYMISHA_INDEXED_TRACK_WRITER_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "TrackIndex.h"

// Indexed 1D-track direct writer (track.dat + track.idx).
//
// The .dat contents are a chromid-ordered concatenation of the per-chrom
// file bytes (signature + payload). Chroms with no payload contribute
// 0 bytes and get length=0 in the index. The .idx file is the 36-byte
// TrackIdxHeader (PMTrackIndexedFormat.cpp) followed by num_chroms
// 24-byte contig entries; CRC64-ECMA is over (chrom_id, offset, length)
// per entry in chromid order. Mirrors the per-chrom + pm_track_convert_to_indexed
// pipeline byte-for-byte. See PMTrackIndexedFormat.cpp.
//
// Usage:
//   IndexedTrackWriter w(track_dir, MishaTrackType::DENSE, num_chroms);
//   for each chrom in chromid order:
//       w.begin_chrom(chromid);
//       w.stream_bytes(&binsize, sizeof(binsize));     // dense header
//       w.stream_bytes(bins.data(), bins.size() * 4);  // payload
//       w.end_chrom();
//   w.finish();
//
// Or, for sparse-with-prebuilt-payload:
//   w.append_chrom(chromid, payload_ptr, payload_len);  // length 0 ok
struct IndexedTrackWriter {
    static constexpr std::size_t HEADER_BYTES = 36;
    static constexpr std::size_t HEADER_TO_CHECKSUM = 28;
    static constexpr std::size_t ENTRY_BYTES = 24;

    std::string dat_path;
    std::string idx_path;
    std::string dat_tmp;
    std::string idx_tmp;
    FILE *dat_fp;
    FILE *idx_fp;
    std::vector<TrackContigEntry> entries;
    std::uint64_t current_offset;
    MishaTrackType track_type;
    std::uint32_t num_chroms;
    bool finished;

    IndexedTrackWriter(const std::string &track_dir, MishaTrackType type,
                       std::uint32_t nchroms);
    ~IndexedTrackWriter();

    IndexedTrackWriter(const IndexedTrackWriter &) = delete;
    IndexedTrackWriter &operator=(const IndexedTrackWriter &) = delete;

    void write_initial_header();
    void append_chrom(int chromid, const void *payload, std::uint64_t length);
    void begin_chrom(int chromid);
    void stream_bytes(const void *data, std::size_t n);
    void end_chrom();
    void finish();
};

#endif
