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
#include <memory>

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
        TGLError<GenomeTrack>("Failed to open %s: %s", src.c_str(), strerror(errno));

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
    std::vector<char> buf_vec(BUFFER_SIZE);
    char *buffer = buf_vec.data();
    uint64_t total_read = 0;

    while (total_read < file_size) {
        size_t to_read = min((uint64_t)BUFFER_SIZE, file_size - total_read);
        size_t read_bytes = fread(buffer, 1, to_read, src_fp);
        if (read_bytes != to_read) {
            fclose(src_fp);
            TGLError<GenomeTrack>("Failed to read from %s", src.c_str());
        }

        size_t written = fwrite(buffer, 1, read_bytes, dest);
        if (written != read_bytes) {
            fclose(src_fp);
            TGLError<GenomeTrack>("Failed to write to track.dat");
        }

        total_read += read_bytes;
    }

    fclose(src_fp);
    bytes_written = file_size;
    return true;
}

// Helper: parse a chrom-pair filename like "1-2" or "chr1-chr2"
// Returns true if filename matches pattern, fills chrom1_name and chrom2_name
// The old index and data of an indexed track set aside by a forced conversion, put back when
// the conversion fails
static void restore_old_index(const string &old_idx_path, const string &old_dat_path)
{
    if (old_idx_path.empty())
        return;
    string dir = old_idx_path.substr(0, old_idx_path.rfind('/'));
    rename(old_dat_path.c_str(), (dir + "/track.dat").c_str());
    rename(old_idx_path.c_str(), (dir + "/track.idx").c_str());
    TrackIndex2D::clear_cache();
}

PyObject *pm_track2d_convert_to_indexed(PyObject *self, PyObject *args)
{
    string dat_path_tmp;
    string idx_path_tmp;
    // the index and data of an indexed track, set aside until the new ones are in place
    string old_idx_path;
    string old_dat_path;

    try {
        PyMisha pymisha(true);

        const char *track_dir_cstr = nullptr;
        int track_type_int = 0;
        int remove_old = 1;
        int force = 0;
        const char *track_cstr = nullptr;
        if (!PyArg_ParseTuple(args, "si|pps", &track_dir_cstr, &track_type_int, &remove_old, &force, &track_cstr)) {
            verror("Invalid arguments to pm_track2d_convert_to_indexed. "
                   "Expected (track_dir: str, track_type: int, remove_old: bool, force: bool, track: str)");
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

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Check if already indexed
        string idx_path = track_dir + "/track.idx";
        string dat_path = track_dir + "/track.dat";
        if (access(idx_path.c_str(), F_OK) == 0) {
            if (!force)
                return PyLong_FromLong(0);  // already indexed

            // Forced (R's gtrack.2d.convert_to_indexed): when no per-pair files are left (removed
            // by an earlier conversion), they are extracted from track.dat, and the pairs are packed
            // again
            string extract_error;
            vector<string> extracted;
            try {
                auto old_idx = TrackIndex2D::get_track_index_2d(track_dir);
                if (old_idx && old_idx->is_loaded()) {
                    // Which pairs have a file: the files the packing below finds, also those named by
                    // aliases (e.g. "2-X" for chr2-chrX)
                    bool have_pair_files = false;
                    string missing_pair_file;
                    const GenomeTrack::ChromFiles pair_files(chromkey, track_dir, true);
                    for (const auto &entry : old_idx->entries()) {
                        if (pair_files.exists(entry.chrom1_id, entry.chrom2_id))
                            have_pair_files = true;
                        else if (missing_pair_file.empty())
                            missing_pair_file = GenomeTrack::get_2d_filename(chromkey, entry.chrom1_id, entry.chrom2_id);
                    }

                    // Only some of the pairs have a file: packing them would replace track.dat, and lose
                    // the data of the other pairs. Refused before anything is changed.
                    if (have_pair_files && !missing_pair_file.empty())
                        extract_error = "Track " + string(track_cstr ? track_cstr : track_dir.c_str()) +
                            " is indexed and has per-pair files for only some of its chromosome pairs (" +
                            missing_pair_file + " is one that is missing): converting it would lose the data of the pairs without a file. "
                            "Remove the per-pair files and run it again to extract all the pairs from track.dat.";

                    if (!have_pair_files) {
                        unique_ptr<FILE, int (*)(FILE *)> dat_fp(fopen(dat_path.c_str(), "rb"), fclose);
                        if (dat_fp) {
                            for (const auto &entry : old_idx->entries()) {
                                string pair_file = track_dir + "/" + GenomeTrack::get_2d_filename(chromkey, entry.chrom1_id, entry.chrom2_id);
                                vector<char> buf(entry.length);
                                if (fseek(dat_fp.get(), entry.offset, SEEK_SET) || fread(buf.data(), 1, entry.length, dat_fp.get()) != entry.length) {
                                    extract_error = "Failed to read " + dat_path;
                                    break;
                                }
                                FILE *pf = fopen(pair_file.c_str(), "wb");
                                if (pf)
                                    extracted.push_back(pair_file);
                                bool written = pf && fwrite(buf.data(), 1, entry.length, pf) == entry.length;
                                if ((pf && fclose(pf)) || !written) {
                                    extract_error = "Failed to write " + pair_file + ": " + strerror(errno);
                                    break;
                                }
                            }
                        }
                    }
                }
            } catch (const bad_alloc &) {
                extract_error = "Out of memory extracting the pairs of " + dat_path;
            } catch (...) {
                // If index is corrupted, just proceed - per-pair files might still exist
                if (!extracted.empty())
                    extract_error = "Failed to extract the pairs of " + dat_path;
            }
            // Pairs cut short or missing would be packed as they are and the old data removed
            if (!extract_error.empty()) {
                for (const string &file : extracted)
                    unlink(file.c_str());
                TGLError<GenomeTrack>("%s", extract_error.c_str());
            }

            // Set the old index and data aside, hidden from the listing; they are put back if the
            // conversion fails
            if (rename(idx_path.c_str(), (track_dir + "/.track.idx.old").c_str()))
                TGLError<GenomeTrack>("Failed to rename %s: %s", idx_path.c_str(), strerror(errno));
            old_idx_path = track_dir + "/.track.idx.old";
            old_dat_path = track_dir + "/.track.dat.old";
            rename(dat_path.c_str(), old_dat_path.c_str());
            TrackIndex2D::clear_cache();
        }

        // The per-pair files, listed once (GenomeTrack::ChromFiles), by the canonical names or by
        // aliases. Not the call's listing: the conversion writes into the directory.
        const GenomeTrack::ChromFiles files(chromkey, track_dir);
        for (const GenomeTrack::ChromFiles::File &file : files.files()) {
            if (!file.link && file.chromid1 < 0)
                verror("Track %s: file %s is not named by a chromosome pair", track_dir.c_str(), file.name.c_str());
        }

        // Sorted by (chromid1, chromid2) for deterministic output; two files of a pair are an error
        const vector<pair<int, int>> chrom_pairs = files.chrom_pairs();

        if (chrom_pairs.empty()) {
            if (!old_idx_path.empty())
                verror("No valid chromosome pair files found in track directory %s", track_dir.c_str());
            // No per-pair files found, return 0
            return PyLong_FromLong(0);
        }

        // Create tmp files
        dat_path_tmp = track_dir + "/track.dat.tmp";
        idx_path_tmp = track_dir + "/track.idx.tmp";

        FILE *dat_fp = fopen(dat_path_tmp.c_str(), "wb");
        if (!dat_fp) {
            TGLError<GenomeTrack>("Failed to create %s: %s", dat_path_tmp.c_str(), strerror(errno));
        }

        // Build entries and concatenate file contents
        vector<Track2DPairEntry> entries;
        uint64_t current_offset = 0;

        for (const auto &chrom_pair : chrom_pairs) {
            Track2DPairEntry idx_entry;
            idx_entry.chrom1_id = (uint32_t)chrom_pair.first;
            idx_entry.chrom2_id = (uint32_t)chrom_pair.second;
            idx_entry.offset = current_offset;
            idx_entry.length = 0;
            idx_entry.reserved = 0;

            uint64_t bytes_written = 0;
            try {
                copy_file_contents_2d(files.path(chrom_pair.first, chrom_pair.second), dat_fp, bytes_written);
            } catch (...) {
                fclose(dat_fp);
                throw;
            }
            idx_entry.length = bytes_written;
            current_offset += bytes_written;

            entries.push_back(idx_entry);
        }

        // Check the write (a full disk shows up at flush or close) and the size
        // BEFORE renaming over anything: until then the old track is intact.
        bool write_ok = fflush(dat_fp) == 0 && fsync(fileno(dat_fp)) == 0;
        int write_errno = errno;
        if (fclose(dat_fp) != 0 && write_ok) {
            write_ok = false;
            write_errno = errno;
        }
        if (!write_ok)
            TGLError<GenomeTrack>("Failed to write %s: %s", dat_path_tmp.c_str(), strerror(write_errno));

        struct stat dat_stat;
        if (stat(dat_path_tmp.c_str(), &dat_stat) != 0) {
            TGLError<GenomeTrack>("Failed to stat %s after conversion", dat_path_tmp.c_str());
        }
        if ((uint64_t)dat_stat.st_size != current_offset) {
            TGLError<GenomeTrack>("track.dat size mismatch: expected %llu bytes, got %llu bytes",
                                  (unsigned long long)current_offset,
                                  (unsigned long long)dat_stat.st_size);
        }

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

        // Remove old per-pair files if requested, also those that the packed ones hid
        if (remove_old) {
            for (const GenomeTrack::ChromFiles::File &file : files.files()) {
                if (!file.link || file.chromid1 >= 0)  // not a symlink the conversion ignored
                    unlink((track_dir + "/" + file.name).c_str());
            }
        }

        if (!old_idx_path.empty()) {
            unlink(old_idx_path.c_str());
            unlink(old_dat_path.c_str());
        }

        // Clear the 2D index cache so it reloads from disk
        TrackIndex2D::clear_cache();

        return PyLong_FromLong((long)entries.size());

    } catch (TGLException &e) {
        unlink(dat_path_tmp.c_str());
        unlink(idx_path_tmp.c_str());
        restore_old_index(old_idx_path, old_dat_path);
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const bad_alloc &) {
        unlink(dat_path_tmp.c_str());
        unlink(idx_path_tmp.c_str());
        restore_old_index(old_idx_path, old_dat_path);
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
