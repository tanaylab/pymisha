#include "IndexedTrackWriter.h"

#include <cerrno>
#include <cstring>
#include <unistd.h>

#include "CRC64.h"
#include "GenomeTrack.h"
#include "TGLException.h"

IndexedTrackWriter::IndexedTrackWriter(const std::string &track_dir,
                                       MishaTrackType type,
                                       std::uint32_t nchroms)
    : dat_path(track_dir + "/track.dat"),
      idx_path(track_dir + "/track.idx"),
      dat_tmp(track_dir + "/track.dat.tmp"),
      idx_tmp(track_dir + "/track.idx.tmp"),
      dat_fp(nullptr), idx_fp(nullptr),
      current_offset(0), track_type(type), num_chroms(nchroms),
      finished(false)
{
    dat_fp = fopen(dat_tmp.c_str(), "wb");
    if (!dat_fp)
        TGLError<GenomeTrack>("Failed to create %s: %s", dat_tmp.c_str(), strerror(errno));
    idx_fp = fopen(idx_tmp.c_str(), "wb");
    if (!idx_fp) {
        fclose(dat_fp); dat_fp = nullptr;
        unlink(dat_tmp.c_str());
        TGLError<GenomeTrack>("Failed to create %s: %s", idx_tmp.c_str(), strerror(errno));
    }
    // Bump stdio buffer to 1 MiB so per-record fwrite()s coalesce into
    // few syscalls. Critical on networked filesystems where the default
    // ~4-8 KiB buffer means thousands of write() syscalls for a 1M-row
    // sparse track (-1.0 s wall on NFS). Best-effort: if setvbuf fails,
    // libc keeps using its default buffer.
    setvbuf(dat_fp, nullptr, _IOFBF, 1 << 20);
    setvbuf(idx_fp, nullptr, _IOFBF, 1 << 20);
    entries.reserve(num_chroms);
    write_initial_header();
}

IndexedTrackWriter::~IndexedTrackWriter()
{
    // Cleanup on early destruction (exception path).
    if (!finished) {
        if (dat_fp) fclose(dat_fp);
        if (idx_fp) fclose(idx_fp);
        unlink(dat_tmp.c_str());
        unlink(idx_tmp.c_str());
    }
}

void IndexedTrackWriter::write_initial_header()
{
    // Matches TrackIdxHeader (PMTrackIndexedFormat.cpp): packed 36 bytes,
    // checksum zero for now and rewritten after entries are computed.
#pragma pack(push, 1)
    struct DiskHeader {
        char magic[8];
        uint32_t version;
        uint32_t track_type_raw;
        uint32_t num_contigs;
        uint64_t flags;
        uint64_t checksum;
    };
#pragma pack(pop)
    DiskHeader hdr;
    const char magic[8] = {'M','I','S','H','A','T','D','X'};
    memcpy(hdr.magic, magic, 8);
    hdr.version = 1;
    hdr.track_type_raw = static_cast<uint32_t>(track_type);
    hdr.num_contigs = num_chroms;
    hdr.flags = 0x01; // IS_LITTLE_ENDIAN
    hdr.checksum = 0;

    if (fwrite(&hdr, sizeof(hdr), 1, idx_fp) != 1)
        TGLError<GenomeTrack>("Failed to write index header to %s", idx_tmp.c_str());
}

void IndexedTrackWriter::append_chrom(int chromid, const void *payload,
                                      std::uint64_t length)
{
    TrackContigEntry entry;
    entry.chrom_id = (uint32_t)chromid;
    entry.offset = current_offset;
    entry.length = length;
    entry.reserved = 0;

    if (length > 0) {
        if (fwrite(payload, 1, length, dat_fp) != length)
            TGLError<GenomeTrack>("Failed to write chrom payload to %s: %s",
                                  dat_tmp.c_str(), strerror(errno));
        current_offset += length;
    }
    entries.push_back(entry);

#pragma pack(push, 1)
    struct DiskContigEntry {
        uint32_t chrom_id;
        uint64_t offset;
        uint64_t length;
        uint32_t reserved;
    };
#pragma pack(pop)
    DiskContigEntry de;
    de.chrom_id = entry.chrom_id;
    de.offset = entry.offset;
    de.length = entry.length;
    de.reserved = entry.reserved;
    if (fwrite(&de, sizeof(de), 1, idx_fp) != 1)
        TGLError<GenomeTrack>("Failed to write index entry to %s", idx_tmp.c_str());
}

void IndexedTrackWriter::begin_chrom(int chromid)
{
    TrackContigEntry entry;
    entry.chrom_id = (uint32_t)chromid;
    entry.offset = current_offset;
    entry.length = 0;
    entry.reserved = 0;
    entries.push_back(entry);
}

void IndexedTrackWriter::stream_bytes(const void *data, std::size_t n)
{
    if (n == 0) return;
    if (fwrite(data, 1, n, dat_fp) != n)
        TGLError<GenomeTrack>("Failed to write chrom payload to %s: %s",
                              dat_tmp.c_str(), strerror(errno));
    current_offset += n;
}

void IndexedTrackWriter::end_chrom()
{
    TrackContigEntry &entry = entries.back();
    entry.length = current_offset - entry.offset;

#pragma pack(push, 1)
    struct DiskContigEntry {
        uint32_t chrom_id;
        uint64_t offset;
        uint64_t length;
        uint32_t reserved;
    };
#pragma pack(pop)
    DiskContigEntry de;
    de.chrom_id = entry.chrom_id;
    de.offset = entry.offset;
    de.length = entry.length;
    de.reserved = entry.reserved;
    if (fwrite(&de, sizeof(de), 1, idx_fp) != 1)
        TGLError<GenomeTrack>("Failed to write index entry to %s", idx_tmp.c_str());
}

void IndexedTrackWriter::finish()
{
    if ((uint32_t)entries.size() != num_chroms)
        TGLError<GenomeTrack>("IndexedTrackWriter: wrote %u entries, expected %u",
                              (unsigned)entries.size(), (unsigned)num_chroms);

    misha::CRC64 crc64;
    uint64_t checksum = crc64.init_incremental();
    for (const auto &entry : entries) {
        checksum = crc64.compute_incremental(checksum,
            (const unsigned char*)&entry.chrom_id, sizeof(entry.chrom_id));
        checksum = crc64.compute_incremental(checksum,
            (const unsigned char*)&entry.offset, sizeof(entry.offset));
        checksum = crc64.compute_incremental(checksum,
            (const unsigned char*)&entry.length, sizeof(entry.length));
    }
    checksum = crc64.finalize_incremental(checksum);

    if (fseek(idx_fp, (long)HEADER_TO_CHECKSUM, SEEK_SET) != 0)
        TGLError<GenomeTrack>("Failed to seek to checksum in %s", idx_tmp.c_str());
    if (fwrite(&checksum, sizeof(checksum), 1, idx_fp) != 1)
        TGLError<GenomeTrack>("Failed to write checksum to %s", idx_tmp.c_str());

    fflush(dat_fp);
    fflush(idx_fp);
    fsync(fileno(dat_fp));
    fsync(fileno(idx_fp));
    fclose(dat_fp); dat_fp = nullptr;
    fclose(idx_fp); idx_fp = nullptr;

    if (rename(dat_tmp.c_str(), dat_path.c_str()) != 0)
        TGLError<GenomeTrack>("Failed to rename %s -> %s: %s",
                              dat_tmp.c_str(), dat_path.c_str(), strerror(errno));
    if (rename(idx_tmp.c_str(), idx_path.c_str()) != 0)
        TGLError<GenomeTrack>("Failed to rename %s -> %s: %s",
                              idx_tmp.c_str(), idx_path.c_str(), strerror(errno));

    finished = true;
}
