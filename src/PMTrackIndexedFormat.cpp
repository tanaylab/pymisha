/*
 * PMTrackIndexedFormat.cpp
 *
 * Convert per-chromosome tracks to indexed format (track.idx/track.dat)
 */

#include <cstdint>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>

#include <Python.h>

#include "PMDb.h"
#include "GenomeTrack.h"
#include "TrackIndex.h"
#include "CRC64.h"
#include "BufferedFile.h"
#include "TGLException.h"
#include "pymisha.h"

using namespace std;

// Offset to checksum field in index header
static const size_t IDX_HEADER_SIZE_TO_CHECKSUM =
    8 +                    // Magic header
    sizeof(uint32_t) +     // Version
    sizeof(uint32_t) +     // Track type
    sizeof(uint32_t) +     // Num contigs
    sizeof(uint64_t);      // Flags

static void write_index_header(FILE *fp, MishaTrackType track_type, uint32_t num_contigs, uint64_t checksum)
{
    const char magic[8] = {'M','I','S','H','A','T','D','X'};
    if (fwrite(magic, 1, 8, fp) != 8)
        TGLError<GenomeTrack>("Failed to write index header");

    uint32_t version = 1;
    if (fwrite(&version, sizeof(version), 1, fp) != 1)
        TGLError<GenomeTrack>("Failed to write index version");

    uint32_t track_type_raw = static_cast<uint32_t>(track_type);
    if (fwrite(&track_type_raw, sizeof(track_type_raw), 1, fp) != 1)
        TGLError<GenomeTrack>("Failed to write track type");

    if (fwrite(&num_contigs, sizeof(num_contigs), 1, fp) != 1)
        TGLError<GenomeTrack>("Failed to write number of contigs");

    uint64_t flags = 0x01; // IS_LITTLE_ENDIAN
    if (fwrite(&flags, sizeof(flags), 1, fp) != 1)
        TGLError<GenomeTrack>("Failed to write flags");

    if (fwrite(&checksum, sizeof(checksum), 1, fp) != 1)
        TGLError<GenomeTrack>("Failed to write checksum");
}

static bool copy_file_contents(const string &src, FILE *dest, uint64_t &bytes_written)
{
    FILE *src_fp = fopen(src.c_str(), "rb");
    if (!src_fp)
        return false;

    if (fseek(src_fp, 0, SEEK_END) != 0) {
        fclose(src_fp);
        TGLError<GenomeTrack>("Failed to seek to end of %s", src.c_str());
    }
    uint64_t file_size = ftello(src_fp);
    if (fseek(src_fp, 0, SEEK_SET) != 0) {
        fclose(src_fp);
        TGLError<GenomeTrack>("Failed to seek to start of %s", src.c_str());
    }

    const size_t BUFFER_SIZE = 1024 * 1024;
    char *buffer = new char[BUFFER_SIZE];
    uint64_t total_read = 0;

    while (total_read < file_size) {
        size_t to_read = min((uint64_t)BUFFER_SIZE, file_size - total_read);
        size_t read_bytes = fread(buffer, 1, to_read, src_fp);
        if (read_bytes != to_read) {
            delete[] buffer;
            fclose(src_fp);
            TGLError<GenomeTrack>("Failed to read from %s", src.c_str());
        }

        size_t written = fwrite(buffer, 1, read_bytes, dest);
        if (written != read_bytes) {
            delete[] buffer;
            fclose(src_fp);
            TGLError<GenomeTrack>("Failed to write to track.dat");
        }

        total_read += read_bytes;
    }

    delete[] buffer;
    fclose(src_fp);
    bytes_written = file_size;
    return true;
}

static MishaTrackType map_track_type(GenomeTrack::Type type)
{
    switch (type) {
        case GenomeTrack::FIXED_BIN:
            return MishaTrackType::DENSE;
        case GenomeTrack::SPARSE:
            return MishaTrackType::SPARSE;
        case GenomeTrack::ARRAYS:
            return MishaTrackType::ARRAY;
        default:
            TGLError<GenomeTrack>("Only 1D tracks (dense, sparse, array) can be converted");
    }
    return MishaTrackType::DENSE;
}

PyObject *pm_track_convert_to_indexed(PyObject *self, PyObject *args)
{
    string dat_path_tmp;
    string idx_path_tmp;

    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        int remove_old = 0;
        if (!PyArg_ParseTuple(args, "s|p", &track, &remove_old)) {
            verror("Invalid arguments to pm_track_convert_to_indexed");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        string track_name(track);
        if (!g_pmdb->track_exists(track_name)) {
            verror("Track not found: %s", track);
        }

        string track_dir = g_pmdb->track_path(track_name);
        string idx_path = track_dir + "/track.idx";
        if (access(idx_path.c_str(), F_OK) == 0) {
            Py_INCREF(Py_None);
            return Py_None; // already indexed
        }

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        GenomeTrack::Type track_type_enum = GenomeTrack::get_type(track_dir.c_str(), chromkey);
        MishaTrackType track_type = map_track_type(track_type_enum);

        dat_path_tmp = track_dir + "/track.dat.tmp";
        idx_path_tmp = track_dir + "/track.idx.tmp";
        string dat_path = track_dir + "/track.dat";

        FILE *dat_fp = fopen(dat_path_tmp.c_str(), "wb");
        if (!dat_fp) {
            TGLError<GenomeTrack>("Failed to create %s: %s", dat_path_tmp.c_str(), strerror(errno));
        }

        FILE *idx_fp = fopen(idx_path_tmp.c_str(), "wb");
        if (!idx_fp) {
            fclose(dat_fp);
            TGLError<GenomeTrack>("Failed to create %s: %s", idx_path_tmp.c_str(), strerror(errno));
        }

        write_index_header(idx_fp, track_type, chromkey.get_num_chroms(), 0);

        vector<TrackContigEntry> entries;
        vector<string> chr_files_to_remove;
        uint64_t current_offset = 0;

        unordered_set<string> existing_files;
        DIR *dir = opendir(track_dir.c_str());
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_name[0] != '.') {
                    existing_files.insert(entry->d_name);
                }
            }
            closedir(dir);
        }

        for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); chromid++) {
            string chrom_name = chromkey.id2chrom(chromid);

            string chr_file;
            bool found = false;

            if (existing_files.count(chrom_name)) {
                chr_file = track_dir + "/" + chrom_name;
                found = true;
            } else if (chrom_name.substr(0, 3) != "chr" && existing_files.count("chr" + chrom_name)) {
                chr_file = track_dir + "/chr" + chrom_name;
                found = true;
            } else if (chrom_name.substr(0, 3) == "chr" && existing_files.count(chrom_name.substr(3))) {
                chr_file = track_dir + "/" + chrom_name.substr(3);
                found = true;
            } else {
                vector<string> aliases;
                chromkey.get_aliases(chromid, aliases);
                for (const auto &alias : aliases) {
                    if (existing_files.count(alias)) {
                        chr_file = track_dir + "/" + alias;
                        found = true;
                        break;
                    }
                }
            }

            TrackContigEntry entry;
            entry.chrom_id = chromid;
            entry.offset = current_offset;
            entry.length = 0;
            entry.reserved = 0;

            uint64_t bytes_written = 0;
            if (found && copy_file_contents(chr_file, dat_fp, bytes_written)) {
                entry.length = bytes_written;
                current_offset += bytes_written;
                chr_files_to_remove.push_back(chr_file);
            }

            if (fwrite(&entry.chrom_id, sizeof(entry.chrom_id), 1, idx_fp) != 1 ||
                fwrite(&entry.offset, sizeof(entry.offset), 1, idx_fp) != 1 ||
                fwrite(&entry.length, sizeof(entry.length), 1, idx_fp) != 1 ||
                fwrite(&entry.reserved, sizeof(entry.reserved), 1, idx_fp) != 1) {
                fclose(dat_fp);
                fclose(idx_fp);
                TGLError<GenomeTrack>("Failed to write index entry for chromosome %s", chrom_name.c_str());
            }

            entries.push_back(entry);
        }

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

        if (fseek(idx_fp, IDX_HEADER_SIZE_TO_CHECKSUM, SEEK_SET) != 0) {
            fclose(dat_fp);
            fclose(idx_fp);
            TGLError<GenomeTrack>("Failed to seek to checksum position in index");
        }
        if (fwrite(&checksum, sizeof(checksum), 1, idx_fp) != 1) {
            fclose(dat_fp);
            fclose(idx_fp);
            TGLError<GenomeTrack>("Failed to update checksum in index");
        }

        fflush(dat_fp);
        fflush(idx_fp);
        fsync(fileno(dat_fp));
        fsync(fileno(idx_fp));

        fclose(dat_fp);
        fclose(idx_fp);

        if (rename(dat_path_tmp.c_str(), dat_path.c_str()) != 0) {
            TGLError<GenomeTrack>("Failed to rename %s to %s: %s",
                                  dat_path_tmp.c_str(), dat_path.c_str(), strerror(errno));
        }
        if (rename(idx_path_tmp.c_str(), idx_path.c_str()) != 0) {
            TGLError<GenomeTrack>("Failed to rename %s to %s: %s",
                                  idx_path_tmp.c_str(), idx_path.c_str(), strerror(errno));
        }

        struct stat dat_stat;
        if (stat(dat_path.c_str(), &dat_stat) != 0) {
            TGLError<GenomeTrack>("Failed to stat %s after conversion", dat_path.c_str());
        }
        if ((uint64_t)dat_stat.st_size != current_offset) {
            TGLError<GenomeTrack>("track.dat size mismatch: expected %llu bytes, got %llu bytes",
                                  (unsigned long long)current_offset, (unsigned long long)dat_stat.st_size);
        }

        if (remove_old) {
            for (const auto &chr_file : chr_files_to_remove) {
                unlink(chr_file.c_str());
            }
        }

    } catch (TGLException &e) {
        unlink(dat_path_tmp.c_str());
        unlink(idx_path_tmp.c_str());
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        unlink(dat_path_tmp.c_str());
        unlink(idx_path_tmp.c_str());
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *pm_track_create_empty_indexed(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        if (!PyArg_ParseTuple(args, "s", &track)) {
            verror("Invalid arguments to pm_track_create_empty_indexed");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        string track_name(track);
        string track_dir = g_pmdb->track_path(track_name);
        string idx_path = track_dir + "/track.idx";
        string dat_path = track_dir + "/track.dat";

        FILE *dat_fp = fopen(dat_path.c_str(), "wb");
        if (!dat_fp) {
            TGLError<GenomeTrack>("Failed to create %s: %s", dat_path.c_str(), strerror(errno));
        }
        fclose(dat_fp);

        FILE *idx_fp = fopen(idx_path.c_str(), "wb");
        if (!idx_fp) {
            unlink(dat_path.c_str());
            TGLError<GenomeTrack>("Failed to create %s: %s", idx_path.c_str(), strerror(errno));
        }

        uint64_t checksum = 0;
        write_index_header(idx_fp, MishaTrackType::SPARSE, 0, checksum);
        fclose(idx_fp);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}
