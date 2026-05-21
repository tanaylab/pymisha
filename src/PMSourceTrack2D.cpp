// pm_read_source_track_2d: C++ port of the 2D source-track reader for
// gtrack_liftover (G1.P3.D). Mirrors the 1D PMSourceTrack but handles
// per-pair quadtree files (RECTS / POINTS) instead of dense/sparse 1D.
//
// Output: SourceTrack2DRows - parallel vectors (chrom1, chrom2, x1, y1, x2, y2, value).
// POINTS are unified to RECTS shape via (x, x+1, y, y+1, val).

#include "pymisha.h"
#include "PMSourceTrack2D.h"
#include "QuadTreeReader.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

bool is_per_pair_candidate(const char *name) {
    if (!name || !*name) return false;
    if (name[0] == '.') return false;
    if (!strcmp(name, "track.idx")) return false;
    if (!strcmp(name, "track.dat")) return false;
    return true;
}

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

// Parse the (chrom1, chrom2) from a per-pair file name. Returns false if the
// name does not contain a single '-' separator we can split on. Empty chrom
// names are also rejected.
//
// Strategy: split on the LAST '-' since some R-style names embed a chrom-side
// 'chr-' prefix (e.g. "chr1-chr2") which contains no '-' itself, but also we
// must handle names like "chrX-chrY" cleanly. Last-occurrence split keeps
// "chr1" / "chr2" intact.
//
// Edge case: "X-Y" - first '-' equals last '-', split returns ("X", "Y").
// Edge case: "1-2-3" (unexpected) - split on last '-' gives ("1-2", "3") which
// is just a string we hand to the chain matcher; if it fails to match it's
// silently dropped downstream. We do NOT special-case this.
bool split_pair_name(const std::string &name, std::string &c1, std::string &c2)
{
    auto pos = name.rfind('-');
    if (pos == std::string::npos) return false;
    if (pos == 0 || pos == name.size() - 1) return false;
    c1 = name.substr(0, pos);
    c2 = name.substr(pos + 1);
    return !c1.empty() && !c2.empty();
}

// Read the root-node arena from a per-pair file buffer.
// Returns false if the buffer is too small / signature invalid.
//
// Buffer layout (from pymisha/_quadtree.py + GenomeTrack.cpp):
//   offset  0: int32 signature   (-9 RECTS, -10 POINTS)
//   offset  4: uint64 num_objs
//   offset 12: int64  root_chunk_fpos
// At root_chunk_fpos:
//   offset 0: int64 chunk_size
//   offset 8: int64 top_node_offset
// At (root_chunk_fpos + top_node_offset): NodeBase containing arena.
bool extract_arena_and_open(
    const std::vector<char> &buf, bool &is_points,
    int64_t &root_chunk_fpos, quadtree::Rectangle &arena)
{
    if (buf.size() < 20) return false;
    int32_t sig;
    std::memcpy(&sig, buf.data(), sizeof(sig));
    if (sig == quadtree::SIGNATURE_RECTS) {
        is_points = false;
    } else if (sig == quadtree::SIGNATURE_POINTS) {
        is_points = true;
    } else {
        return false;  // Not a 2D file (signature mismatch); caller skips it.
    }

    uint64_t num_objs = 0;
    quadtree::parse_header(buf.data(), buf.size(), num_objs, root_chunk_fpos);
    if (num_objs == 0) {
        arena = {0, 0, 0, 0};  // Empty pair, sentinel.
        return true;
    }

    int64_t top_node_offset = quadtree::get_chunk_top_node_offset(buf.data(), root_chunk_fpos);
    int64_t arena_offset    = root_chunk_fpos + top_node_offset;
    if (arena_offset < 0 || (size_t)arena_offset + sizeof(quadtree::NodeBase) > buf.size()) {
        return false;
    }
    const quadtree::NodeBase *nb = reinterpret_cast<const quadtree::NodeBase *>(buf.data() + arena_offset);
    arena = nb->arena;
    return true;
}

void append_rect(SourceTrack2DRows &out, const std::string &c1, const std::string &c2,
                 int64_t x1, int64_t y1, int64_t x2, int64_t y2, double v)
{
    // Skip zero-length rects to match R (which never emits such rectangles
    // from a valid track but we filter defensively).
    if (x1 >= x2 || y1 >= y2) return;
    out.chrom1.emplace_back(c1);
    out.chrom2.emplace_back(c2);
    out.x1.push_back(x1);
    out.y1.push_back(y1);
    out.x2.push_back(x2);
    out.y2.push_back(y2);
    out.value.push_back(v);
}

}  // namespace

void read_source_track_2d_cpp(
    const std::string &src_track_dir,
    SourceTrack2DRows &out)
{
    struct stat st;
    if (stat(src_track_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        throw std::runtime_error("Source track 2D directory does not exist: " + src_track_dir);
    }

    std::vector<std::string> entries;
    if (!list_dir_sorted(src_track_dir, entries)) {
        throw std::runtime_error("Cannot list source track directory: " + src_track_dir);
    }

    bool any_seen = false;
    bool seen_is_points = false;
    bool got_pair = false;  // Track whether any valid per-pair file was opened.

    for (const auto &name : entries) {
        if (!is_per_pair_candidate(name.c_str())) continue;

        std::string path = src_track_dir + "/" + name;
        struct stat fst;
        if (stat(path.c_str(), &fst) != 0 || !S_ISREG(fst.st_mode)) continue;

        std::vector<char> buf;
        if (!slurp_file(path, buf)) {
            throw std::runtime_error("Failed to read file: " + path);
        }

        bool file_is_points = false;
        int64_t root_chunk_fpos = 0;
        quadtree::Rectangle arena{};
        if (!extract_arena_and_open(buf, file_is_points, root_chunk_fpos, arena)) {
            // Not a 2D quadtree file (e.g. 1D source dense/sparse coexisting).
            // Skip silently; the caller's auto-detect ensures we wouldn't be
            // called on a mixed-purpose directory in production.
            continue;
        }

        if (got_pair && file_is_points != seen_is_points) {
            throw std::invalid_argument(
                "Mixed RECTS / POINTS 2D source files in directory: " + src_track_dir);
        }
        seen_is_points = file_is_points;
        got_pair = true;
        any_seen = true;

        std::string c1, c2;
        if (!split_pair_name(name, c1, c2)) {
            // Not a valid pair name; skip (could be a stray non-pair file).
            continue;
        }

        // Empty pair (num_objs == 0) - extract_arena_and_open returned a
        // zero arena. Nothing to emit.
        if (arena.x2 <= arena.x1 || arena.y2 <= arena.y1) {
            continue;
        }

        // Walk the quadtree with a full-arena query to enumerate every object.
        quadtree::QueryObjects qo = quadtree::query_objects(
            buf.data(), buf.size(),
            file_is_points,
            root_chunk_fpos,
            arena.x1, arena.y1, arena.x2, arena.y2,
            /*band=*/nullptr);

        for (size_t i = 0; i < qo.ids.size(); ++i) {
            append_rect(out, c1, c2,
                        qo.x1s[i], qo.y1s[i], qo.x2s[i], qo.y2s[i],
                        (double)qo.vals[i]);
        }
    }

    out.is_points = seen_is_points;
    (void)any_seen;  // Sentinel for future debug; unused now.
}

// =========================================================================
// Python entry point
// =========================================================================

PyObject *pm_read_source_track_2d(PyObject *self, PyObject *args)
{
    const char *src_track_dir = nullptr;
    if (!PyArg_ParseTuple(args, "s", &src_track_dir)) return nullptr;

    try {
        SourceTrack2DRows rows;
        read_source_track_2d_cpp(src_track_dir, rows);

        npy_intp n = (npy_intp)rows.chrom1.size();
        PMPY py_c1(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
        PMPY py_c2(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
        PMPY py_x1(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
        PMPY py_y1(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
        PMPY py_x2(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
        PMPY py_y2(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
        PMPY py_v (PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
        if (!py_c1 || !py_c2 || !py_x1 || !py_y1 || !py_x2 || !py_y2 || !py_v) return nullptr;

        PyObject **c1_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_c1);
        PyObject **c2_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_c2);
        for (npy_intp i = 0; i < n; ++i) {
            Py_INCREF(Py_None); c1_out[i] = Py_None;
            Py_INCREF(Py_None); c2_out[i] = Py_None;
        }
        int64_t *x1o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_x1);
        int64_t *y1o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_y1);
        int64_t *x2o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_x2);
        int64_t *y2o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_y2);
        double  *vo  = (double *) PyArray_DATA((PyArrayObject *)*py_v);

        for (npy_intp i = 0; i < n; ++i) {
            PyObject *s1 = PyUnicode_FromStringAndSize(
                rows.chrom1[i].data(), (Py_ssize_t)rows.chrom1[i].size());
            PyObject *s2 = PyUnicode_FromStringAndSize(
                rows.chrom2[i].data(), (Py_ssize_t)rows.chrom2[i].size());
            if (!s1 || !s2) { Py_XDECREF(s1); Py_XDECREF(s2); return nullptr; }
            Py_DECREF(c1_out[i]); c1_out[i] = s1;
            Py_DECREF(c2_out[i]); c2_out[i] = s2;
            x1o[i] = rows.x1[i]; y1o[i] = rows.y1[i];
            x2o[i] = rows.x2[i]; y2o[i] = rows.y2[i];
            vo[i]  = rows.value[i];
        }

        PMPY result(PyDict_New(), true);
        if (!result) return nullptr;
        if (PyDict_SetItemString(*result, "chrom1", *py_c1) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "chrom2", *py_c2) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "x1",     *py_x1) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "y1",     *py_y1) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "x2",     *py_x2) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "y2",     *py_y2) < 0) return nullptr;
        if (PyDict_SetItemString(*result, "value",  *py_v)  < 0) return nullptr;
        PyObject *ip = PyBool_FromLong(rows.is_points ? 1 : 0);
        if (PyDict_SetItemString(*result, "is_points", ip) < 0) { Py_DECREF(ip); return nullptr; }
        Py_DECREF(ip);

        result.to_be_stolen();
        return (PyObject *)*result;

    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    }
}
