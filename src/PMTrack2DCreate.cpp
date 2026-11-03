// The writer of gtrack_2d_create.
//
// Port of R misha 5.12.1's gtrack_create_track2d (GenomeTrack2DCreate.cpp),
// written so that the track files are the bytes R writes for the same input: the
// objects go into R's quad trees (StatQuadTree.h, StatQuadTreeCached.h and
// StatQuadTreeCachedSerializer.h, shared with R unchanged) in the order R inserts
// them, and the trees are written in chunks of R's gtrack.chunk.size.
//
// Python reads and checks the input; this function gets the records as
// chromosome ids, coordinates and float values. The 2D imports are in
// PMTrack2DImport.cpp.

#include "pymisha.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "BufferedFile.h"
#include "PMTrackExpression2DIterator.h"
#include "StatQuadTreeCached.h"
#include "StatQuadTreeCachedSerializer.h"
#include "TGLException.h"

namespace {

// R misha's gtrack.chunk.size and gtrack.num.chunks defaults (ConfigurationDefaults.h)
const int64_t TRACK_CHUNK_SIZE = 100000;
const int64_t TRACK_NUM_CHUNKS = 0;

// GenomeTrack::FORMAT_SIGNATURES of RECTS and POINTS
const int32_t RECTS_SIGNATURE = -9;
const int32_t POINTS_SIGNATURE = -10;

PyArrayObject *get_array(PyObject *obj, const char *what, int dtype, npy_intp n)
{
    PyObject *arr = PyArray_FROM_OTF(obj, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        verror("%s has a wrong type", what);
    if (PyArray_NDIM((PyArrayObject *)arr) != 1 || (n >= 0 && PyArray_DIM((PyArrayObject *)arr, 0) != n)) {
        Py_DECREF(arr);
        verror("%s must be 1-D and as long as the other columns", what);
    }
    return (PyArrayObject *)arr;
}

// The records of a call: columns of one length, owned for the call's duration.
struct Records {
    std::vector<PMPY> owners;
    const int32_t *chromid1 = nullptr, *chromid2 = nullptr;
    const int64_t *x1 = nullptr, *x2 = nullptr, *y1 = nullptr, *y2 = nullptr;
    const float *val = nullptr;
    npy_intp n = 0;

    template <class T>
    const T *column(PyObject *obj, const char *what, int dtype) {
        PyArrayObject *arr = get_array(obj, what, dtype, owners.empty() ? -1 : n);
        if (owners.empty())
            n = PyArray_DIM(arr, 0);
        owners.emplace_back((PyObject *)arr, true);
        return (const T *)PyArray_DATA(arr);
    }
};

struct Genome {
    std::vector<std::string> chroms;
    std::vector<int64_t>     sizes;

    void read(PyObject *chroms_obj, PyObject *sizes_obj) {
        PMPY seq(PySequence_Fast(chroms_obj, "chroms"), true);
        if (!seq)
            verror("chroms must be a sequence");
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*seq); ++i) {
            const char *s = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(*seq, i));
            if (!s)
                verror("chroms must be strings");
            chroms.emplace_back(s);
        }
        PMPY sseq(PySequence_Fast(sizes_obj, "sizes"), true);
        if (!sseq || PySequence_Fast_GET_SIZE(*sseq) != (Py_ssize_t)chroms.size())
            verror("sizes must hold a size per chromosome");
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*sseq); ++i) {
            long long size = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(*sseq, i));
            if (size == -1 && PyErr_Occurred())
                verror("sizes must be integers");
            sizes.push_back(size);
        }
    }

    void check(int chromid) const {
        if (chromid < 0 || chromid >= (int)chroms.size())
            verror("Invalid chromosome id %d", chromid);
    }

    // the file of a chromosome pair, named as pymisha names it
    std::string filename(const std::string &dir, int chromid1, int chromid2) const {
        return dir + "/" + chroms[chromid1] + "-" + chroms[chromid2];
    }
};

void open_chk(BufferedFile &file, const std::string &path, const char *mode)
{
    if (file.open(path.c_str(), mode))
        TGLError("Opening file %s: %s", path.c_str(), strerror(errno));
}

void write_signature(BufferedFile &file, const std::string &path, int32_t signature)
{
    open_chk(file, path, "wb");
    if (file.write(&signature, sizeof(signature)) != sizeof(signature))
        TGLError("Failed to write to file %s: %s", path.c_str(), strerror(errno));
}

void close_chk(BufferedFile &file)
{
    if (file.close())
        TGLError("Failed to write to file %s: %s", file.file_name().c_str(), strerror(errno));
}

} // namespace

// pm_track_2d_create(track_dir, chroms, sizes, chromid1, start1, end1, chromid2, start2, end2, values)
//
// gtrack_create_track2d: the intervals sorted as R sorts them (r_2d_sort_order), each pair
// inserted into a RectsQuadTree in that order and written as a RECTS file, whatever the
// size of the intervals. values: float32. The caller has checked the intervals, overlaps
// included.
PyObject *pm_track_2d_create(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *dir_str;
        PyObject *chroms_obj, *sizes_obj, *c1, *s1, *e1, *c2, *s2, *e2, *v;
        if (!PyArg_ParseTuple(args, "sOOOOOOOOO", &dir_str, &chroms_obj, &sizes_obj, &c1, &s1, &e1, &c2, &s2, &e2, &v))
            return nullptr;

        Genome genome;
        genome.read(chroms_obj, sizes_obj);
        Records r;
        r.chromid1 = r.column<int32_t>(c1, "chromid1", NPY_INT32);
        r.x1 = r.column<int64_t>(s1, "start1", NPY_INT64);
        r.x2 = r.column<int64_t>(e1, "end1", NPY_INT64);
        r.chromid2 = r.column<int32_t>(c2, "chromid2", NPY_INT32);
        r.y1 = r.column<int64_t>(s2, "start2", NPY_INT64);
        r.y2 = r.column<int64_t>(e2, "end2", NPY_INT64);
        r.val = r.column<float>(v, "values", NPY_FLOAT32);

        std::vector<GInterval2D> intervs(r.n);
        for (npy_intp i = 0; i < r.n; ++i) {
            genome.check(r.chromid1[i]);
            genome.check(r.chromid2[i]);
            intervs[i] = GInterval2D(r.chromid1[i], r.x1[i], r.x2[i], r.chromid2[i], r.y1[i], r.y2[i]);
        }
        std::vector<size_t> order = r_2d_sort_order(intervs);

        const std::string dirname(dir_str);
        RectsQuadTree qtree;
        for (size_t first = 0, last; first < order.size(); first = last) {
            const GInterval2D &f = intervs[order[first]];
            qtree.reset(0, 0, genome.sizes[f.chromid1], genome.sizes[f.chromid2]);
            for (last = first; last < order.size() && intervs[order[last]].is_same_chrom_pair(f); ++last) {
                const GInterval2D &iv = intervs[order[last]];
                qtree.insert(RectsQuadTree::ValueType(iv.start1, iv.start2, iv.end1, iv.end2, r.val[order[last]]));
                check_interrupt();
            }

            BufferedFile file;
            write_signature(file, genome.filename(dirname, f.chromid1, f.chromid2), RECTS_SIGNATURE);
            RectsQuadTreeCached qtree_cached(TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);
            qtree_cached.serialize(file, qtree);
            close_chk(file);
        }
        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}
