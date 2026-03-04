/*
 * PMTrack2DIndexedFormat.cpp
 *
 * Convert per-pair 2D tracks to indexed format (track.idx/track.dat)
 * and query index info for 2D tracks.
 */

#include <cstdint>
#include <dirent.h>
#include <errno.h>
#include <regex.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

#include <Python.h>

#include "PMDb.h"
#include "GenomeTrack.h"
#include "TrackIndex2D.h"
#include "CRC64.h"
#include "TGLException.h"
#include "pymisha.h"

using namespace std;

// Helper: copy file contents to destination, returning bytes written
static bool copy_file_contents_2d(const string &src, FILE *dest, uint64_t &bytes_written)
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

// Helper: parse a chrom-pair filename like "1-2" or "chr1-chr2"
// Returns true if filename matches pattern, fills chrom1_name and chrom2_name
static bool parse_pair_filename(const string &filename, string &chrom1_name, string &chrom2_name)
{
    // Find the last dash that separates the two chrom names
    // Since chrom names can contain digits but not dashes, find '-' as separator
    // Pattern: name1-name2 where name1 and name2 don't contain '-'
    // But chrom names can look like "chr1", "1", "X", "chr10", etc.
    // We need to find the right '-' separator.
    // For simplicity: try splitting on each '-' and see if both parts look like chrom names

    // Skip files that are obviously not pair files
    if (filename == "track.idx" || filename == "track.dat" ||
        filename == "track.idx.tmp" || filename == "track.dat.tmp" ||
        filename == ".attributes") {
        return false;
    }

    // Find the separator dash
    // Strategy: the filename is "c1-c2" where c1 and c2 are chrom names.
    // Chrom names don't contain dashes in standard genomes, so split on first dash.
    size_t dash_pos = filename.find('-');
    if (dash_pos == string::npos || dash_pos == 0 || dash_pos == filename.size() - 1) {
        return false;
    }

    chrom1_name = filename.substr(0, dash_pos);
    chrom2_name = filename.substr(dash_pos + 1);

    // Both parts must be non-empty
    if (chrom1_name.empty() || chrom2_name.empty()) {
        return false;
    }

    return true;
}

// Helper: strip "chr" prefix if present
static string strip_chr_prefix(const string &name)
{
    if (name.size() > 3 && name.substr(0, 3) == "chr") {
        return name.substr(3);
    }
    return name;
}

PyObject *pm_track2d_convert_to_indexed(PyObject *self, PyObject *args)
{
    string dat_path_tmp;
    string idx_path_tmp;

    try {
        PyMisha pymisha(true);

        const char *track_dir_cstr = nullptr;
        int track_type_int = 0;
        if (!PyArg_ParseTuple(args, "si", &track_dir_cstr, &track_type_int)) {
            verror("Invalid arguments to pm_track2d_convert_to_indexed. "
                   "Expected (track_dir: str, track_type: int)");
        }

        if (track_type_int != 0 && track_type_int != 1) {
            verror("track_type must be 0 (RECTS) or 1 (POINTS), got %d", track_type_int);
        }

        MishaTrack2DType track_type = static_cast<MishaTrack2DType>(track_type_int);
        string track_dir(track_dir_cstr);

        // Check track_dir exists
        struct stat dir_stat;
        if (stat(track_dir.c_str(), &dir_stat) != 0 || !S_ISDIR(dir_stat.st_mode)) {
            verror("Track directory does not exist: %s", track_dir.c_str());
        }

        // Check if already indexed
        string idx_path = track_dir + "/track.idx";
        if (access(idx_path.c_str(), F_OK) == 0) {
            // Already indexed, return 0
            return PyLong_FromLong(0);
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Enumerate all files in track_dir that look like per-pair files
        DIR *dir = opendir(track_dir.c_str());
        if (!dir) {
            verror("Failed to open track directory %s: %s", track_dir.c_str(), strerror(errno));
        }

        // Collect pair files: map from (chrom1_id, chrom2_id) -> filepath
        struct PairFile {
            uint32_t chrom1_id;
            uint32_t chrom2_id;
            string filepath;
        };
        vector<PairFile> pair_files;

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;

            string filename(entry->d_name);
            string chrom1_name, chrom2_name;

            if (!parse_pair_filename(filename, chrom1_name, chrom2_name)) {
                continue;
            }

            // Try to resolve chrom names to IDs
            // Try as-is first, then strip chr prefix
            int chrom1_id = chromkey.chrom2id(chrom1_name.c_str());
            if (chrom1_id < 0) {
                string stripped = strip_chr_prefix(chrom1_name);
                chrom1_id = chromkey.chrom2id(stripped.c_str());
            }

            int chrom2_id = chromkey.chrom2id(chrom2_name.c_str());
            if (chrom2_id < 0) {
                string stripped = strip_chr_prefix(chrom2_name);
                chrom2_id = chromkey.chrom2id(stripped.c_str());
            }

            if (chrom1_id < 0 || chrom2_id < 0) {
                continue;  // Skip files with unrecognizable chrom names
            }

            PairFile pf;
            pf.chrom1_id = (uint32_t)chrom1_id;
            pf.chrom2_id = (uint32_t)chrom2_id;
            pf.filepath = track_dir + "/" + filename;
            pair_files.push_back(pf);
        }
        closedir(dir);

        if (pair_files.empty()) {
            // No per-pair files found, return 0
            return PyLong_FromLong(0);
        }

        // Sort by (chrom1_id, chrom2_id)
        sort(pair_files.begin(), pair_files.end(),
            [](const PairFile &a, const PairFile &b) {
                if (a.chrom1_id != b.chrom1_id) return a.chrom1_id < b.chrom1_id;
                return a.chrom2_id < b.chrom2_id;
            });

        // Create tmp files
        dat_path_tmp = track_dir + "/track.dat.tmp";
        idx_path_tmp = track_dir + "/track.idx.tmp";
        string dat_path = track_dir + "/track.dat";

        FILE *dat_fp = fopen(dat_path_tmp.c_str(), "wb");
        if (!dat_fp) {
            TGLError<GenomeTrack>("Failed to create %s: %s", dat_path_tmp.c_str(), strerror(errno));
        }

        // Build entries and concatenate file contents
        vector<Track2DPairEntry> entries;
        vector<string> files_to_remove;
        uint64_t current_offset = 0;

        for (const auto &pf : pair_files) {
            Track2DPairEntry idx_entry;
            idx_entry.chrom1_id = pf.chrom1_id;
            idx_entry.chrom2_id = pf.chrom2_id;
            idx_entry.offset = current_offset;
            idx_entry.length = 0;
            idx_entry.reserved = 0;

            uint64_t bytes_written = 0;
            if (copy_file_contents_2d(pf.filepath, dat_fp, bytes_written)) {
                idx_entry.length = bytes_written;
                current_offset += bytes_written;
                files_to_remove.push_back(pf.filepath);
            }

            entries.push_back(idx_entry);
        }

        fflush(dat_fp);
        fsync(fileno(dat_fp));
        fclose(dat_fp);

        // Write index file via TrackIndex2D
        TrackIndex2D::write_index(idx_path_tmp, track_type, entries);

        // Atomic rename
        if (rename(dat_path_tmp.c_str(), dat_path.c_str()) != 0) {
            TGLError<GenomeTrack>("Failed to rename %s to %s: %s",
                                  dat_path_tmp.c_str(), dat_path.c_str(), strerror(errno));
        }
        if (rename(idx_path_tmp.c_str(), idx_path.c_str()) != 0) {
            TGLError<GenomeTrack>("Failed to rename %s to %s: %s",
                                  idx_path_tmp.c_str(), idx_path.c_str(), strerror(errno));
        }

        // Verify track.dat size
        struct stat dat_stat;
        if (stat(dat_path.c_str(), &dat_stat) != 0) {
            TGLError<GenomeTrack>("Failed to stat %s after conversion", dat_path.c_str());
        }
        if ((uint64_t)dat_stat.st_size != current_offset) {
            TGLError<GenomeTrack>("track.dat size mismatch: expected %llu bytes, got %llu bytes",
                                  (unsigned long long)current_offset,
                                  (unsigned long long)dat_stat.st_size);
        }

        // Remove old per-pair files
        for (const auto &f : files_to_remove) {
            unlink(f.c_str());
        }

        // Clear the 2D index cache so it reloads from disk
        TrackIndex2D::clear_cache();

        return PyLong_FromLong((long)entries.size());

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
}

PyObject *pm_track2d_index_info(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track_dir_cstr = nullptr;
        if (!PyArg_ParseTuple(args, "s", &track_dir_cstr)) {
            verror("Invalid arguments to pm_track2d_index_info. Expected (track_dir: str)");
        }

        string track_dir(track_dir_cstr);
        string idx_path = track_dir + "/track.idx";

        // Build result dict
        PyObject *result = PyDict_New();
        if (!result) {
            verror("Failed to create result dictionary");
        }

        // Try to load the index
        TrackIndex2D idx;
        bool loaded = idx.load(idx_path);

        PyDict_SetItemString(result, "loaded", loaded ? Py_True : Py_False);

        if (!loaded) {
            // Return minimal info
            PyDict_SetItemString(result, "track_type", Py_None);
            PyObject *zero = PyLong_FromLong(0);
            PyDict_SetItemString(result, "num_pairs", zero);
            Py_DECREF(zero);
            PyObject *empty_list = PyList_New(0);
            PyDict_SetItemString(result, "pairs", empty_list);
            Py_DECREF(empty_list);
            return result;
        }

        // Track type
        const char *type_str = (idx.get_track_type() == MishaTrack2DType::RECTS) ? "RECTS" : "POINTS";
        PyObject *py_type = PyUnicode_FromString(type_str);
        PyDict_SetItemString(result, "track_type", py_type);
        Py_DECREF(py_type);

        // Num pairs
        PyObject *py_num = PyLong_FromLong((long)idx.num_entries());
        PyDict_SetItemString(result, "num_pairs", py_num);
        Py_DECREF(py_num);

        // Pairs list
        const auto &entries = idx.entries();
        PyObject *pairs_list = PyList_New(entries.size());
        if (!pairs_list) {
            Py_DECREF(result);
            verror("Failed to create pairs list");
        }

        for (size_t i = 0; i < entries.size(); ++i) {
            const auto &e = entries[i];
            PyObject *pair_dict = PyDict_New();
            if (!pair_dict) {
                Py_DECREF(pairs_list);
                Py_DECREF(result);
                verror("Failed to create pair dictionary");
            }

            PyObject *py_c1 = PyLong_FromUnsignedLong(e.chrom1_id);
            PyObject *py_c2 = PyLong_FromUnsignedLong(e.chrom2_id);
            PyObject *py_off = PyLong_FromUnsignedLongLong(e.offset);
            PyObject *py_len = PyLong_FromUnsignedLongLong(e.length);

            PyDict_SetItemString(pair_dict, "chrom1_id", py_c1);
            PyDict_SetItemString(pair_dict, "chrom2_id", py_c2);
            PyDict_SetItemString(pair_dict, "offset", py_off);
            PyDict_SetItemString(pair_dict, "length", py_len);

            Py_DECREF(py_c1);
            Py_DECREF(py_c2);
            Py_DECREF(py_off);
            Py_DECREF(py_len);

            PyList_SET_ITEM(pairs_list, i, pair_dict);
        }

        PyDict_SetItemString(result, "pairs", pairs_list);
        Py_DECREF(pairs_list);

        return result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}
