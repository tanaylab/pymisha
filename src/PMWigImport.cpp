// pm_parse_wig_or_bedgraph: streaming C++ parser for UCSC WIG and BedGraph.
//
// Mirrors pymisha.tracks._parse_wig_or_bedgraph but bypasses the Python-level
// line loop and per-row list appends. Plain-text only; the Python wrapper
// handles gzipped paths by falling back to its pure-Python streamer.

#include "pymisha.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
            if (*endp) verror("Malformed %s line at %s:%ld - invalid start=%s",
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
 *
 * Raises pymisha.error on malformed input or empty file.
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

    // Reserve modestly; we'll grow as needed. Most real files are 10^5-10^7 rows.
    chrom.reserve(1u << 12);
    start.reserve(1u << 12);
    end.reserve(1u << 12);
    value.reserve(1u << 12);

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

            if (mode == MODE_FIXED && ntok == 1) {
                char *endp = nullptr;
                double v = strtod(toks[0], &endp);
                if (*endp) verror("Cannot parse WIG value at %s:%ld - '%s'", path, lineno, toks[0]);
                chrom.push_back(cur_chrom);
                start.push_back(cur_pos0);
                end.push_back(cur_pos0 + cur_span);
                value.push_back(v);
                cur_pos0 += cur_step;
                continue;
            }
            if (mode == MODE_VAR && ntok >= 2) {
                char *endp = nullptr;
                double pos_f = strtod(toks[0], &endp);
                if (*endp) verror("Cannot parse variableStep position at %s:%ld - '%s'", path, lineno, toks[0]);
                int64_t pos0 = (int64_t)pos_f - 1; // WIG 1-based
                double v = strtod(toks[1], &endp);
                if (*endp) verror("Cannot parse variableStep value at %s:%ld - '%s'", path, lineno, toks[1]);
                chrom.push_back(cur_chrom);
                start.push_back(pos0);
                end.push_back(pos0 + cur_span);
                value.push_back(v);
                continue;
            }
            if (ntok >= 4) {
                // BedGraph: chrom start end value
                char *endp = nullptr;
                double s_f = strtod(toks[1], &endp);
                if (*endp) verror("Cannot parse BedGraph start at %s:%ld - '%s'", path, lineno, toks[1]);
                double e_f = strtod(toks[2], &endp);
                if (*endp) verror("Cannot parse BedGraph end at %s:%ld - '%s'", path, lineno, toks[2]);
                double v = strtod(toks[3], &endp);
                if (*endp) verror("Cannot parse BedGraph value at %s:%ld - '%s'", path, lineno, toks[3]);
                chrom.push_back(std::string(toks[0]));
                start.push_back((int64_t)s_f);
                end.push_back((int64_t)e_f);
                value.push_back(v);
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

    if (chrom.empty()) {
        PyErr_Format(PyExc_ValueError, "WIG/BedGraph file '%s' contains no intervals", path);
        return nullptr;
    }

    npy_intp n = (npy_intp)chrom.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_value(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);

    if (!py_chrom || !py_start || !py_end || !py_value) {
        return nullptr;
    }

    PyObject **chrom_data = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    int64_t *start_data = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_data = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    double *value_data = (double *)PyArray_DATA((PyArrayObject *)*py_value);

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(chrom[i].data(), chrom[i].size());
        if (!s) return nullptr;
        chrom_data[i] = s;
        start_data[i] = start[i];
        end_data[i] = end[i];
        value_data[i] = value[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return nullptr;
    PyDict_SetItemString(result, "chrom", py_chrom);
    PyDict_SetItemString(result, "start", py_start);
    PyDict_SetItemString(result, "end", py_end);
    PyDict_SetItemString(result, "value", py_value);

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
