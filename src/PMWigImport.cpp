// pm_parse_wig_or_bedgraph: streaming C++ parser for UCSC WIG and BedGraph.
//
// Plain-text only: gtrack_import unzips a .gz / .zip file first, as misha does.
//
// pm_is_bigwig / pm_parse_bigwig / pm_parse_bigbed: misha's native bigWig and bigBed reader (BigWig.cpp).

#include "pymisha.h"
#include "BigWig.h"
#include "PMDb.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// PMStubs.cpp
extern void convert_py_intervals(PyObject *py_intervals, std::vector<GInterval> &intervals);
extern void intervals_sort(std::vector<GInterval> &intervals);
extern void intervals_unify_overlaps(std::vector<GInterval> &intervals, bool merge_touching);

namespace {

enum Mode { MODE_NONE, MODE_FIXED, MODE_VAR };

// Parse "k=v" tokens in a fixedStep/variableStep declaration line into the
// requested keys. Unknown keys are ignored. Throws via verror on parse error.
struct StepDecl {
    std::string chrom;
    int64_t start = -1;
    int64_t step = 1;
    int64_t span = 1;
    bool has_chrom = false;
    bool has_start = false;
};

bool starts_with_ci(const char *s, const char *prefix)
{
    while (*prefix) {
        if (!*s) return false;
        char a = (char)::tolower((unsigned char)*s);
        char b = (char)::tolower((unsigned char)*prefix);
        if (a != b) return false;
        ++s; ++prefix;
    }
    return true;
}

void parse_kv_args(const char *line, size_t header_skip, StepDecl &decl,
                   const char *kind, long lineno, const char *path)
{
    const char *p = line + header_skip;
    while (*p) {
        // skip whitespace
        while (*p && isspace((unsigned char)*p)) ++p;
        if (!*p) break;
        const char *k_begin = p;
        while (*p && *p != '=' && !isspace((unsigned char)*p)) ++p;
        if (*p != '=') {
            // bare token, skip
            continue;
        }
        std::string key(k_begin, p - k_begin);
        ++p; // skip '='
        const char *v_begin = p;
        while (*p && !isspace((unsigned char)*p)) ++p;
        std::string val(v_begin, p - v_begin);

        // Case-insensitive key match
        for (auto &c : key) c = (char)::tolower((unsigned char)c);
        if (key == "chrom") { decl.chrom = val; decl.has_chrom = true; }
        else if (key == "start") {
            char *endp = nullptr;
            long long n = strtoll(val.c_str(), &endp, 10);
            if (*endp || n < 1) verror("Malformed %s line at %s:%ld - invalid start=%s",
                                         kind, path, lineno, val.c_str());
            decl.start = (int64_t)n;
            decl.has_start = true;
        }
        else if (key == "step") {
            char *endp = nullptr;
            long long n = strtoll(val.c_str(), &endp, 10);
            if (*endp) verror("Malformed %s line at %s:%ld - invalid step=%s",
                               kind, path, lineno, val.c_str());
            decl.step = (int64_t)n;
        }
        else if (key == "span") {
            char *endp = nullptr;
            long long n = strtoll(val.c_str(), &endp, 10);
            if (*endp) verror("Malformed %s line at %s:%ld - invalid span=%s",
                               kind, path, lineno, val.c_str());
            decl.span = (int64_t)n;
        }
        // other keys ignored
    }
}

// Strip leading whitespace and return true if line is blank or a comment / header.
// A coordinate as R's Wig::str2coord reads it: an integer >= base, made 0-based.
// (Parsing it as a double let "nan" through as INT64_MIN and "0.9" as 0.)
int64_t wig_coord(const char *str, int64_t base, const char *what, const char *path, long lineno)
{
    char *endp = nullptr;
    long long c = strtoll(str, &endp, 10);
    if (endp == str || *endp || c < base)
        verror("Cannot parse %s at %s:%ld - '%s'", what, path, lineno, str);
    return (int64_t)c - base;
}

// a + b for a, b >= 0, capped at INT64_MAX (R's Wig.cpp add_capped): a record that far is past
// the chromosome end and dropped
inline int64_t add_capped(int64_t a, int64_t b)
{
    return b > INT64_MAX - a ? INT64_MAX : a + b;
}

bool is_skip_line(const char *line)
{
    while (*line && isspace((unsigned char)*line)) ++line;
    if (!*line) return true;
    if (*line == '#') return true;
    if (starts_with_ci(line, "track")) {
        // require following whitespace or end-of-line to avoid skipping a chrom
        // named "track..."
        const char *p = line + 5;
        return (!*p || isspace((unsigned char)*p));
    }
    if (starts_with_ci(line, "browser")) {
        const char *p = line + 7;
        return (!*p || isspace((unsigned char)*p));
    }
    return false;
}

// Split line by whitespace into up to max_tokens null-terminated cstrs.
// Returns the token count. Tokens point into `buf` which is mutated in-place.
size_t split_ws(char *buf, char *out_tokens[], size_t max_tokens)
{
    size_t n = 0;
    char *p = buf;
    while (*p && n < max_tokens) {
        while (*p && isspace((unsigned char)*p)) *p++ = '\0';
        if (!*p) break;
        out_tokens[n++] = p;
        while (*p && !isspace((unsigned char)*p)) ++p;
        if (*p) *p++ = '\0';
    }
    return n;
}

} // namespace

/*
 * pm_parse_wig_or_bedgraph(path) -> dict
 *
 * path:  str - filesystem path to a plain-text WIG or BedGraph file.
 *
 * Returns: dict with keys
 *   - "chrom": numpy object array of chrom strings
 *   - "start": numpy int64 array (0-based, inclusive)
 *   - "end":   numpy int64 array (exclusive)
 *   - "value": numpy float64 array
 *   - "step":  numpy bool array, true for a fixedStep / variableStep record, false
 *              for bedGraph
 *   - "line":  numpy int64 array, the line of the record (1-based)
 *
 * A file with no records gives empty arrays. Raises pymisha.error on malformed input.
 */
extern PyObject *s_pm_err;

PyObject *pm_parse_wig_or_bedgraph(PyObject *self, PyObject *args)
{
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return nullptr;
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        PyErr_Format(PyExc_OSError, "Cannot open %s: %s", path, strerror(errno));
        return nullptr;
    }

    try {

    std::vector<std::string> chrom;
    std::vector<int64_t> start;
    std::vector<int64_t> end;
    std::vector<double> value;
    std::vector<char> step;
    std::vector<int64_t> lines;

    // Reserve modestly; we'll grow as needed. Most real files are 10^5-10^7 rows.
    chrom.reserve(1u << 12);
    start.reserve(1u << 12);
    end.reserve(1u << 12);
    value.reserve(1u << 12);
    step.reserve(1u << 12);
    lines.reserve(1u << 12);

    Mode mode = MODE_NONE;
    std::string cur_chrom;
    int64_t cur_step = 1;
    int64_t cur_span = 1;
    int64_t cur_pos0 = 0;

    char *line = nullptr;
    size_t cap = 0;
    ssize_t nread = 0;
    long lineno = 0;

    try {
        while ((nread = getline(&line, &cap, fp)) != -1) {
            ++lineno;
            // strip trailing \r\n
            while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r')) {
                line[--nread] = '\0';
            }
            if (nread == 0) continue;
            if (is_skip_line(line)) continue;

            // skip leading whitespace before keyword check
            char *l = line;
            while (*l && isspace((unsigned char)*l)) ++l;

            if (starts_with_ci(l, "fixedstep")) {
                StepDecl decl;
                parse_kv_args(l, 9, decl, "fixedStep", lineno, path);
                if (!decl.has_chrom || !decl.has_start) {
                    verror("Malformed fixedStep line at %s:%ld - missing chrom or start", path, lineno);
                }
                mode = MODE_FIXED;
                cur_chrom = decl.chrom;
                cur_step = decl.step;
                cur_span = decl.span;
                cur_pos0 = decl.start - 1; // WIG uses 1-based starts
                continue;
            }
            if (starts_with_ci(l, "variablestep")) {
                StepDecl decl;
                parse_kv_args(l, 12, decl, "variableStep", lineno, path);
                if (!decl.has_chrom) {
                    verror("Malformed variableStep line at %s:%ld - missing chrom", path, lineno);
                }
                mode = MODE_VAR;
                cur_chrom = decl.chrom;
                cur_span = decl.span;
                continue;
            }

            // Data line. Split.
            char *toks[16];
            size_t ntok = split_ws(l, toks, 16);
            if (ntok == 0) continue;

            if (ntok >= 4) {
                // BedGraph: chrom start end value. As in misha, a bedGraph line is one whatever
                // section it follows, and it ends that section.
                char *endp = nullptr;
                int64_t s_i = wig_coord(toks[1], 0, "BedGraph start", path, lineno);
                int64_t e_i = wig_coord(toks[2], 0, "BedGraph end", path, lineno);
                double v = strtod(toks[3], &endp);
                if (*endp) verror("Cannot parse BedGraph value at %s:%ld - '%s'", path, lineno, toks[3]);
                chrom.push_back(std::string(toks[0]));
                start.push_back(s_i);
                end.push_back(e_i);
                value.push_back(v);
                step.push_back(0);
                lines.push_back(lineno);
                mode = MODE_NONE;
                continue;
            }
            if (ntok == 1 && mode != MODE_FIXED)
                verror("Invalid format of WIG file %s, line %ld: value is not preceded by fixedStep header", path, lineno);
            if (ntok == 2 && mode != MODE_VAR)
                verror("Invalid format of WIG file %s, line %ld: value is not preceded by variableStep header", path, lineno);
            if (mode == MODE_FIXED && ntok == 1) {
                char *endp = nullptr;
                double v = strtod(toks[0], &endp);
                if (*endp) verror("Cannot parse WIG value at %s:%ld - '%s'", path, lineno, toks[0]);
                chrom.push_back(cur_chrom);
                start.push_back(cur_pos0);
                end.push_back(add_capped(cur_pos0, cur_span));
                value.push_back(v);
                step.push_back(1);
                lines.push_back(lineno);
                cur_pos0 = add_capped(cur_pos0, cur_step);
                continue;
            }
            if (mode == MODE_VAR && ntok >= 2) {
                char *endp = nullptr;
                int64_t pos0 = wig_coord(toks[0], 1, "variableStep position", path, lineno); // WIG 1-based
                double v = strtod(toks[1], &endp);
                if (*endp) verror("Cannot parse variableStep value at %s:%ld - '%s'", path, lineno, toks[1]);
                chrom.push_back(cur_chrom);
                start.push_back(pos0);
                end.push_back(add_capped(pos0, cur_span));
                value.push_back(v);
                step.push_back(1);
                lines.push_back(lineno);
                continue;
            }
            verror("Cannot parse WIG/BedGraph line at %s:%ld - '%s'", path, lineno, l);
        }
        if (ferror(fp)) {
            verror("Error reading %s: %s", path, strerror(errno));
        }
    } catch (...) {
        free(line);
        fclose(fp);
        throw;
    }
    free(line);
    fclose(fp);

    npy_intp n = (npy_intp)chrom.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_value(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
    PMPY py_step(PyArray_SimpleNew(1, &n, NPY_BOOL), true);
    PMPY py_line(PyArray_SimpleNew(1, &n, NPY_INT64), true);

    if (!py_chrom || !py_start || !py_end || !py_value || !py_step || !py_line) {
        return nullptr;
    }

    PyObject **chrom_data = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    int64_t *start_data = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_data = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    double *value_data = (double *)PyArray_DATA((PyArrayObject *)*py_value);
    npy_bool *step_data = (npy_bool *)PyArray_DATA((PyArrayObject *)*py_step);
    int64_t *line_data = (int64_t *)PyArray_DATA((PyArrayObject *)*py_line);

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(chrom[i].data(), chrom[i].size());
        if (!s) return nullptr;
        chrom_data[i] = s;
        start_data[i] = start[i];
        end_data[i] = end[i];
        value_data[i] = value[i];
        step_data[i] = step[i];
        line_data[i] = lines[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return nullptr;
    PyDict_SetItemString(result, "chrom", py_chrom);
    PyDict_SetItemString(result, "start", py_start);
    PyDict_SetItemString(result, "end", py_end);
    PyDict_SetItemString(result, "value", py_value);
    PyDict_SetItemString(result, "step", py_step);
    PyDict_SetItemString(result, "line", py_line);

    return_py(result);

    } catch (TGLException &e) {
        // pymisha.error, as the docstring above promises. This catch was dead until
        // verror() was made non-returning: a malformed file used to parse past the
        // bad line and return a dict with a Python error already set.
        PyErr_SetString(s_pm_err, e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return_err();
    }

    return_none();
}

// pm_is_bigwig(path) -> bool: misha's test, the bigWig magic number or a .bw / .bigWig name.
PyObject *pm_is_bigwig(PyObject *self, PyObject *args)
{
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return nullptr;
    return PyBool_FromLong(BigWig::is_bigwig(path));
}

static PyObject *string_list(const std::vector<std::string> &strs)
{
    PyObject *list = PyList_New(strs.size());
    if (!list)
        return nullptr;
    for (size_t i = 0; i < strs.size(); ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(strs[i].data(), strs[i].size());
        if (!s) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, s);
    }
    return list;
}

// The database names of the chromosomes of intervals, as a numpy object array: one string object
// per chromosome, shared by its rows. nullptr on error.
static PyObject *chrom_array(const GIntervals &intervals, const GenomeChromKey &chromkey)
{
    npy_intp n = (npy_intp)intervals.size();
    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    if (!py_chrom)
        return nullptr;

    std::vector<PMPY> names(chromkey.get_num_chroms());
    PyObject **chrom_data = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    for (npy_intp i = 0; i < n; ++i) {
        PMPY &name = names[intervals[i].chromid];
        if (!name) {
            const string &s = chromkey.id2chrom(intervals[i].chromid);
            name.assign(PyUnicode_FromStringAndSize(s.data(), s.size()), true);
            if (!name)
                return nullptr;
        }
        Py_INCREF((PyObject *)name);
        chrom_data[i] = name;
    }
    py_chrom.to_be_stolen();
    return py_chrom;
}

// Fills result with chrom, start and end of intervals and the reports of the reader (see
// pm_parse_bigwig). false on error.
static bool fill_result(PyObject *result, const GIntervals &intervals, const BigWig &reader, const GenomeChromKey &chromkey)
{
    npy_intp n = (npy_intp)intervals.size();
    PMPY py_chrom(chrom_array(intervals, chromkey), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    if (!py_chrom || !py_start || !py_end)
        return false;
    for (npy_intp i = 0; i < n; ++i) {
        ((int64_t *)PyArray_DATA((PyArrayObject *)*py_start))[i] = intervals[i].start;
        ((int64_t *)PyArray_DATA((PyArrayObject *)*py_end))[i] = intervals[i].end;
    }

    const UnknownChroms &unknown = reader.get_unknown_chroms();
    PMPY py_unknown(string_list(unknown.names()), true);
    PMPY py_primary(string_list(unknown.primary_names()), true);
    PMPY py_num_unknown(PyLong_FromUnsignedLongLong(unknown.num()), true);
    PMPY py_truncated(PyBool_FromLong(unknown.truncated()), true);
    PMPY py_num_primary(PyLong_FromUnsignedLongLong(unknown.num_primary()), true);
    PMPY py_num_matched(PyLong_FromUnsignedLongLong(reader.get_num_matched_chroms()), true);
    PMPY py_num_clipped(PyLong_FromUnsignedLongLong(reader.get_num_clipped()), true);
    PMPY py_num_dropped(PyLong_FromUnsignedLongLong(reader.get_num_dropped()), true);
    const string &first_clipped = reader.get_first_clipped();
    PMPY py_first_clipped(PyUnicode_DecodeUTF8(first_clipped.data(), first_clipped.size(), "replace"), true);
    if (!py_unknown || !py_primary || !py_num_unknown || !py_truncated || !py_num_primary || !py_num_matched || !py_num_clipped ||
        !py_num_dropped || !py_first_clipped)
        return false;

    PyDict_SetItemString(result, "chrom", py_chrom);
    PyDict_SetItemString(result, "start", py_start);
    PyDict_SetItemString(result, "end", py_end);
    PyDict_SetItemString(result, "unknown", py_unknown);
    PyDict_SetItemString(result, "num_unknown", py_num_unknown);
    PyDict_SetItemString(result, "truncated", py_truncated);
    PyDict_SetItemString(result, "unknown_primary", py_primary);
    PyDict_SetItemString(result, "num_primary", py_num_primary);
    PyDict_SetItemString(result, "num_matched", py_num_matched);
    PyDict_SetItemString(result, "num_clipped", py_num_clipped);
    PyDict_SetItemString(result, "num_dropped", py_num_dropped);
    PyDict_SetItemString(result, "first_clipped", py_first_clipped);
    return true;
}

/*
 * pm_parse_bigwig(path, intervals=None) -> dict
 *
 * Reads the full-resolution records of a bigWig with misha's reader (BigWig.cpp), with its checks:
 * records are sorted and non-overlapping, a record that runs past the end of its chromosome is
 * clipped to it (dropped if it starts at or past the end), zero-length records are skipped, a
 * truncated or corrupt file is an error. With intervals (the _intervals_to_cpp format), only the
 * records that overlap them are returned, each once and whole, and only the data blocks the index
 * places over them are read; only those records are counted as clipped or dropped.
 *
 * Returns: dict with keys
 *   - "chrom", "start", "end", "value": the records, sorted, chrom holding database chromosome names
 *   - "unknown": up to 5 names of bigWig chromosomes with data that the database lacks
 *   - "num_unknown": the number of such names
 *   - "truncated": whether num_unknown is a lower bound (see UnknownChroms)
 *   - "unknown_primary": up to 5 of them that look like a primary chromosome
 *   - "num_primary": the number of those
 *   - "num_matched": the number of database chromosomes the file has data for
 *   - "num_clipped", "num_dropped": the records past the end of their chromosome that were clipped,
 *     and dropped
 *   - "first_clipped": the first of those in the file, "(chrom, start, end), chromosome size N"
 *     ("" if none)
 */
PyObject *pm_parse_bigwig(PyObject *self, PyObject *args)
{
    const char *path;
    PyObject *py_intervals = Py_None;
    if (!PyArg_ParseTuple(args, "s|O", &path, &py_intervals))
        return nullptr;

    try {
        PyMisha pymisha(true);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        bool all = py_intervals == Py_None;
        BigWig bigwig;
        GIntervals query, chrom_query, data, res;

        bigwig.init(chromkey, path);
        if (!all) {
            convert_py_intervals(py_intervals, query);
            intervals_sort(query);
            intervals_unify_overlaps(query, true);
        }

        GIntervals::const_iterator iquery = query.begin();
        for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
            if (all) {
                for (uint64_t iblock = 0; bigwig.get_data(chromid, iblock, data); ++iblock) {
                    res.insert(res.end(), data.begin(), data.end());
                    check_interrupt();
                }
            } else {
                chrom_query.clear();
                for (; iquery != query.end() && iquery->chromid == chromid; ++iquery)
                    chrom_query.push_back(*iquery);
                if (!chrom_query.empty()) {
                    bigwig.get_data(chromid, chrom_query, data);
                    res.insert(res.end(), data.begin(), data.end());
                }
            }
            check_interrupt();
        }

        npy_intp n = (npy_intp)res.size();
        PMPY py_value(PyArray_SimpleNew(1, &n, NPY_FLOAT32), true);
        PMPY result(PyDict_New(), true);
        if (!py_value || !result || !fill_result(result, res, bigwig, chromkey))
            return_err();
        for (npy_intp i = 0; i < n; ++i)
            memcpy((float *)PyArray_DATA((PyArrayObject *)*py_value) + i, &res[i].udata, sizeof(float));
        PyDict_SetItemString(result, "value", py_value);
        return_py(result);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

/*
 * pm_parse_bigbed(path) -> dict
 *
 * Reads all the records of a bigBed with misha's reader (BigWig.cpp), in file order. Records on
 * chromosomes the database lacks are skipped, records past the end of their chromosome are clipped
 * or dropped as by pm_parse_bigwig.
 *
 * Returns: dict with the keys of pm_parse_bigwig except "value", and
 *   - "fields": the names of the fields past chrom, start and end, from the file's autoSql (or the
 *     BED standard names if it has none)
 *   - "columns": their values, one numpy array per field: float64 for a numeric autoSql type,
 *     object (str) otherwise
 */
PyObject *pm_parse_bigbed(PyObject *self, PyObject *args)
{
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return nullptr;

    try {
        PyMisha pymisha(true);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        BigWig bigbed;
        BigWig::BedRecords records;

        bigbed.init(chromkey, path, true);
        bigbed.read_bed(records);

        const std::vector<BigWig::Field> &fields = bigbed.get_fields();
        npy_intp n = (npy_intp)records.intervals.size();
        std::vector<std::string> names;
        PMPY py_columns(PyList_New(fields.size()), true);
        PMPY result(PyDict_New(), true);
        if (!py_columns || !result || !fill_result(result, records.intervals, bigbed, chromkey))
            return_err();

        for (size_t ifield = 0; ifield < fields.size(); ++ifield) {
            PyObject *col;

            names.push_back(fields[ifield].name);
            if (fields[ifield].numeric) {
                col = PyArray_SimpleNew(1, &n, NPY_FLOAT64);
                if (col && n)
                    memcpy(PyArray_DATA((PyArrayObject *)col), records.values[ifield].data(), n * sizeof(double));
            } else {
                col = PyArray_SimpleNew(1, &n, NPY_OBJECT);
                if (col) {
                    PyObject **data = (PyObject **)PyArray_DATA((PyArrayObject *)col);
                    for (npy_intp i = 0; i < n; ++i) {
                        const std::string &str = records.texts[ifield][i];
                        data[i] = PyUnicode_DecodeUTF8(str.data(), str.size(), "replace");
                        if (!data[i]) {
                            Py_DECREF(col);
                            col = nullptr;
                            break;
                        }
                    }
                }
            }
            if (!col)
                return_err();
            PyList_SET_ITEM((PyObject *)py_columns, ifield, col);
            check_interrupt();
        }

        PMPY py_fields(string_list(names), true);
        if (!py_fields)
            return_err();
        PyDict_SetItemString(result, "fields", py_fields);
        PyDict_SetItemString(result, "columns", py_columns);
        return_py(result);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}
