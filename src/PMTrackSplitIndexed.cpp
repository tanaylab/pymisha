/*
 * PMTrackSplitIndexed.cpp
 *
 * Splits a 1D indexed-format track (track.dat + track.idx) back into
 * per-chromosome files in the same directory, and the inverse pack
 * operation that takes explicit args (track_dir + chrom_names + type)
 * so it works without active GROOT context. Used by gtrack_copy for
 * cross-db format conversion. Mirrors R misha 5.6.28 commit 062e80e7.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

#include <Python.h>

#include "CRC64.h"
#include "TGLException.h"
#include "TrackIndex.h"
#include "pymisha.h"

using namespace std;

// Offset to checksum field in index header (must match PMTrackIndexedFormat.cpp).
static const size_t IDX_HEADER_SIZE_TO_CHECKSUM =
    8 +                    // Magic header
    sizeof(uint32_t) +     // Version
    sizeof(uint32_t) +     // Track type
    sizeof(uint32_t) +     // Num contigs
    sizeof(uint64_t);      // Flags


PyObject *pm_track_split_indexed_to_per_chrom(PyObject *self, PyObject *args)
{
    vector<string> tmp_files_to_cleanup;
    try {
        PyMisha pymisha(true);

        const char *track_dir_c = nullptr;
        PyObject *py_chrom_names = nullptr;
        int remove_indexed = 0;
        if (!PyArg_ParseTuple(args, "sO|p", &track_dir_c, &py_chrom_names, &remove_indexed)) {
            verror("Invalid arguments to pm_track_split_indexed_to_per_chrom");
        }

        if (!PySequence_Check(py_chrom_names)) {
            verror("chrom_names must be a sequence of strings");
        }
        Py_ssize_t n_chroms = PySequence_Size(py_chrom_names);
        vector<string> chrom_names((size_t)n_chroms);
        for (Py_ssize_t i = 0; i < n_chroms; ++i) {
            PyObject *item = PySequence_GetItem(py_chrom_names, i);
            if (!item || !PyUnicode_Check(item)) {
                Py_XDECREF(item);
                verror("chrom_names[%zd] is not a string", i);
            }
            const char *s = PyUnicode_AsUTF8(item);
            if (!s) {
                Py_DECREF(item);
                verror("chrom_names[%zd] is not valid UTF-8", i);
            }
            chrom_names[(size_t)i] = s;
            Py_DECREF(item);
        }

        const string track_dir(track_dir_c);
        const string idx_path = track_dir + "/track.idx";
        const string dat_path = track_dir + "/track.dat";

        TrackIndex idx;
        if (!idx.load(idx_path))
            verror("track.idx not found in %s", track_dir.c_str());

        FILE *dat_fp = fopen(dat_path.c_str(), "rb");
        if (!dat_fp)
            verror("Failed to open %s: %s", dat_path.c_str(), strerror(errno));

        const size_t BUF = 1 << 20; // 1 MiB
        vector<char> buffer(BUF);

        for (const TrackContigEntry &entry : idx.get_all_entries()) {
            if (entry.chrom_id >= (uint32_t)n_chroms) {
                fclose(dat_fp);
                verror("track.idx references chrom_id %u but only %zd chrom names supplied "
                       "(internal mismatch or corrupt index)",
                       entry.chrom_id, n_chroms);
            }

            const string out_path     = track_dir + "/" + chrom_names[entry.chrom_id];
            const string out_path_tmp = out_path + ".tmp";
            tmp_files_to_cleanup.push_back(out_path_tmp);

            FILE *out_fp = fopen(out_path_tmp.c_str(), "wb");
            if (!out_fp) {
                fclose(dat_fp);
                verror("Failed to create %s: %s", out_path_tmp.c_str(), strerror(errno));
            }

            // Length=0 entries arise when the source had no per-chrom file at convert time.
            // We still touch an output file (atomic via tmp+rename) so that downstream
            // per-chrom invariants hold, but we write zero bytes.
            if (entry.length > 0) {
                if (fseeko(dat_fp, (off_t)entry.offset, SEEK_SET) != 0) {
                    fclose(out_fp); fclose(dat_fp);
                    verror("Failed to seek to offset %llu in %s",
                           (unsigned long long)entry.offset, dat_path.c_str());
                }

                uint64_t remaining = entry.length;
                while (remaining > 0) {
                    size_t to_read = (size_t)std::min((uint64_t)BUF, remaining);
                    size_t got = fread(buffer.data(), 1, to_read, dat_fp);
                    if (got != to_read) {
                        fclose(out_fp); fclose(dat_fp);
                        verror("Short read from %s at offset %llu",
                               dat_path.c_str(), (unsigned long long)entry.offset);
                    }
                    if (fwrite(buffer.data(), 1, got, out_fp) != got) {
                        fclose(out_fp); fclose(dat_fp);
                        verror("Failed to write %s: %s", out_path_tmp.c_str(), strerror(errno));
                    }
                    remaining -= got;
                }
            }

            // Per-file fsync + atomic rename: each per-chrom file is canonical db state;
            // accept N fsyncs (one per contig) for crash-safe individual files.
            fflush(out_fp);
            fsync(fileno(out_fp));
            fclose(out_fp);

            if (rename(out_path_tmp.c_str(), out_path.c_str()) != 0) {
                fclose(dat_fp);
                verror("Failed to rename %s to %s: %s",
                       out_path_tmp.c_str(), out_path.c_str(), strerror(errno));
            }
            tmp_files_to_cleanup.pop_back(); // succeeded
        }

        fclose(dat_fp);

        if (remove_indexed) {
            unlink(dat_path.c_str());
            unlink(idx_path.c_str());
        }

        Py_RETURN_NONE;
    } catch (TGLException &e) {
        for (const string &p : tmp_files_to_cleanup) unlink(p.c_str());
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        for (const string &p : tmp_files_to_cleanup) unlink(p.c_str());
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return NULL;
    }
}


// Pack per-chromosome files in track_dir into track.dat + track.idx.
// Mirrors pm_track_convert_to_indexed but takes explicit args (track_dir,
// chrom_names, track_type) so it works without active GROOT/chromkey.
PyObject *pm_track_pack_per_chrom_to_indexed(PyObject *self, PyObject *args)
{
    string dat_path_tmp;
    string idx_path_tmp;
    try {
        PyMisha pymisha(true);

        const char *track_dir_c = nullptr;
        PyObject *py_chrom_names = nullptr;
        const char *type_str_c = nullptr;
        if (!PyArg_ParseTuple(args, "sOs", &track_dir_c, &py_chrom_names, &type_str_c)) {
            verror("Invalid arguments to pm_track_pack_per_chrom_to_indexed");
        }

        if (!PySequence_Check(py_chrom_names)) {
            verror("chrom_names must be a sequence of strings");
        }
        Py_ssize_t n_chroms = PySequence_Size(py_chrom_names);
        vector<string> chrom_names((size_t)n_chroms);
        for (Py_ssize_t i = 0; i < n_chroms; ++i) {
            PyObject *item = PySequence_GetItem(py_chrom_names, i);
            if (!item || !PyUnicode_Check(item)) {
                Py_XDECREF(item);
                verror("chrom_names[%zd] is not a string", i);
            }
            const char *s = PyUnicode_AsUTF8(item);
            if (!s) {
                Py_DECREF(item);
                verror("chrom_names[%zd] is not valid UTF-8", i);
            }
            chrom_names[(size_t)i] = s;
            Py_DECREF(item);
        }

        const string track_dir(track_dir_c);
        const string type_str(type_str_c);
        MishaTrackType track_type = MishaTrackType::DENSE;
        if (type_str == "dense")       track_type = MishaTrackType::DENSE;
        else if (type_str == "sparse") track_type = MishaTrackType::SPARSE;
        else if (type_str == "array")  track_type = MishaTrackType::ARRAY;
        else verror("Unsupported track_type '%s'; expected dense/sparse/array", type_str.c_str());

        dat_path_tmp = track_dir + "/track.dat.tmp";
        idx_path_tmp = track_dir + "/track.idx.tmp";
        const string dat_path = track_dir + "/track.dat";
        const string idx_path = track_dir + "/track.idx";

        FILE *dat_fp = fopen(dat_path_tmp.c_str(), "wb");
        if (!dat_fp)
            verror("Failed to create %s: %s", dat_path_tmp.c_str(), strerror(errno));
        FILE *idx_fp = fopen(idx_path_tmp.c_str(), "wb");
        if (!idx_fp) {
            fclose(dat_fp);
            verror("Failed to create %s: %s", idx_path_tmp.c_str(), strerror(errno));
        }

        // Header (checksum=0; updated at end). Layout matches write_index_header
        // in PMTrackIndexedFormat.cpp.
        const char magic[8] = {'M','I','S','H','A','T','D','X'};
        const uint32_t version = 1;
        const uint32_t track_type_raw = static_cast<uint32_t>(track_type);
        const uint32_t num_contigs = (uint32_t)n_chroms;
        const uint64_t flags = 0x01; // IS_LITTLE_ENDIAN
        uint64_t checksum_placeholder = 0;
        bool header_ok =
            fwrite(magic, 1, 8, idx_fp) == 8 &&
            fwrite(&version, sizeof(version), 1, idx_fp) == 1 &&
            fwrite(&track_type_raw, sizeof(track_type_raw), 1, idx_fp) == 1 &&
            fwrite(&num_contigs, sizeof(num_contigs), 1, idx_fp) == 1 &&
            fwrite(&flags, sizeof(flags), 1, idx_fp) == 1 &&
            fwrite(&checksum_placeholder, sizeof(checksum_placeholder), 1, idx_fp) == 1;
        if (!header_ok) {
            fclose(dat_fp); fclose(idx_fp);
            verror("Failed to write index header");
        }

        vector<TrackContigEntry> entries;
        vector<string> chr_files_to_remove;
        uint64_t current_offset = 0;

        const size_t BUF = 1 << 20;
        vector<char> buffer(BUF);

        for (int chromid = 0; chromid < n_chroms; ++chromid) {
            const string chr_file = track_dir + "/" + chrom_names[chromid];

            TrackContigEntry entry;
            entry.chrom_id = (uint32_t)chromid;
            entry.offset = current_offset;
            entry.length = 0;
            entry.reserved = 0;

            FILE *src_fp = fopen(chr_file.c_str(), "rb");
            if (src_fp) {
                if (fseeko(src_fp, 0, SEEK_END) != 0) {
                    fclose(src_fp); fclose(dat_fp); fclose(idx_fp);
                    verror("Failed to size %s", chr_file.c_str());
                }
                const uint64_t file_size = (uint64_t)ftello(src_fp);
                rewind(src_fp);

                uint64_t remaining = file_size;
                while (remaining > 0) {
                    size_t to_read = (size_t)std::min((uint64_t)BUF, remaining);
                    size_t got = fread(buffer.data(), 1, to_read, src_fp);
                    if (got != to_read) {
                        fclose(src_fp); fclose(dat_fp); fclose(idx_fp);
                        verror("Short read from %s", chr_file.c_str());
                    }
                    if (fwrite(buffer.data(), 1, got, dat_fp) != got) {
                        fclose(src_fp); fclose(dat_fp); fclose(idx_fp);
                        verror("Failed to write track.dat");
                    }
                    remaining -= got;
                }
                fclose(src_fp);
                entry.length = file_size;
                current_offset += file_size;
                chr_files_to_remove.push_back(chr_file);
            }
            // else: entry stays length=0, no per-chrom file present.

            if (fwrite(&entry.chrom_id,  sizeof(entry.chrom_id),  1, idx_fp) != 1 ||
                fwrite(&entry.offset,    sizeof(entry.offset),    1, idx_fp) != 1 ||
                fwrite(&entry.length,    sizeof(entry.length),    1, idx_fp) != 1 ||
                fwrite(&entry.reserved,  sizeof(entry.reserved),  1, idx_fp) != 1) {
                fclose(dat_fp); fclose(idx_fp);
                verror("Failed to write index entry for %s", chrom_names[chromid].c_str());
            }
            entries.push_back(entry);
        }

        // Compute and patch checksum
        misha::CRC64 crc64;
        uint64_t checksum = crc64.init_incremental();
        for (const auto &e : entries) {
            checksum = crc64.compute_incremental(checksum, (const unsigned char*)&e.chrom_id, sizeof(e.chrom_id));
            checksum = crc64.compute_incremental(checksum, (const unsigned char*)&e.offset,   sizeof(e.offset));
            checksum = crc64.compute_incremental(checksum, (const unsigned char*)&e.length,   sizeof(e.length));
        }
        checksum = crc64.finalize_incremental(checksum);

        if (fseek(idx_fp, (long)IDX_HEADER_SIZE_TO_CHECKSUM, SEEK_SET) != 0) {
            fclose(dat_fp); fclose(idx_fp);
            verror("Failed to seek to checksum position");
        }
        if (fwrite(&checksum, sizeof(checksum), 1, idx_fp) != 1) {
            fclose(dat_fp); fclose(idx_fp);
            verror("Failed to update checksum");
        }

        fflush(dat_fp); fflush(idx_fp);
        fsync(fileno(dat_fp)); fsync(fileno(idx_fp));
        fclose(dat_fp); fclose(idx_fp);

        if (rename(dat_path_tmp.c_str(), dat_path.c_str()) != 0)
            verror("Failed to rename %s to %s: %s", dat_path_tmp.c_str(), dat_path.c_str(), strerror(errno));
        if (rename(idx_path_tmp.c_str(), idx_path.c_str()) != 0)
            verror("Failed to rename %s to %s: %s", idx_path_tmp.c_str(), idx_path.c_str(), strerror(errno));

        // Validate track.dat size matches what we wrote, before destroying source files.
        struct stat dat_stat;
        if (stat(dat_path.c_str(), &dat_stat) != 0)
            verror("Failed to stat %s after pack: %s", dat_path.c_str(), strerror(errno));
        if ((uint64_t)dat_stat.st_size != current_offset)
            verror("track.dat size mismatch after pack: expected %llu bytes, got %llu bytes",
                   (unsigned long long)current_offset,
                   (unsigned long long)dat_stat.st_size);

        // Note: per-chrom files are NOT removed. pymisha's set_vars dispatcher
        // currently checks per-chrom-file existence before delegating to the
        // GenomeTrack reader, so tracks still need per-chrom files alongside
        // track.idx + track.dat to be readable. This mirrors the convention
        // that pm_track_convert_to_indexed uses (remove_old=False by default).
        // The unused `chr_files_to_remove` list is retained for symmetry with
        // the R kernel and as a hook for a future read-path fix.
        (void)chr_files_to_remove;

        Py_RETURN_NONE;
    } catch (TGLException &e) {
        if (!dat_path_tmp.empty()) unlink(dat_path_tmp.c_str());
        if (!idx_path_tmp.empty()) unlink(idx_path_tmp.c_str());
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        if (!dat_path_tmp.empty()) unlink(dat_path_tmp.c_str());
        if (!idx_path_tmp.empty()) unlink(idx_path_tmp.c_str());
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return NULL;
    }
}
