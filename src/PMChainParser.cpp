// pm_parse_chain_file: C++ port of liftover._parse_chain_file.
//
// Stream-parses a UCSC chain file and accumulates ChainInterval records into
// nine numpy arrays. Target chromosomes are resolved against the current
// database's GenomeChromKey: unknown targets are skipped silently. Optional
// `min_score` filtering drops chains below the threshold during the stream so
// that skipped blocks are never materialized.
//
// Overlap-policy handling (handle_src_overlaps / handle_tgt_overlaps) is NOT
// done here - it stays in the Python wrapper `gintervals_load_chain` until
// G1.P4 ports it.

#include "pymisha.h"

#include "GenomeChromKey.h"
#include "PMDb.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Split a line into whitespace-separated tokens, mutating buf in place.
// Returns the token count. Tokens are null-terminated cstrs pointing into buf.
size_t split_ws(char *buf, char *out_tokens[], size_t max_tokens)
{
    size_t n = 0;
    char *p = buf;
    while (*p && n < max_tokens) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
            *p++ = '\0';
        }
        if (!*p) break;
        out_tokens[n++] = p;
        while (*p && !(*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
            ++p;
        }
        if (*p) *p++ = '\0';
    }
    return n;
}

} // namespace

PyObject *pm_parse_chain_file(PyObject *self, PyObject *args)
{
    const char *path;
    double min_score;
    if (!PyArg_ParseTuple(args, "sd", &path, &min_score)) {
        return nullptr;
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        PyErr_Format(PyExc_FileNotFoundError, "Chain file does not exist: %s", path);
        return nullptr;
    }

    // NOTE: per-field validation below uses PyErr_Format(PyExc_ValueError, ...)
    // + manual cleanup_line() + fclose(fp) instead of the codebase's verror()
    // helper. verror() throws TGLException which the outer catch converts to
    // RuntimeError, but pymisha/liftover.py raises ValueError / FileNotFoundError
    // - we preserve those exception types here for drop-in parity.
    try {

    if (!g_pmdb || !g_pmdb->is_initialized()) {
        fclose(fp);
        PyErr_SetString(PyExc_RuntimeError, "Database not initialized. Call gdb_init() first.");
        return nullptr;
    }
    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    std::vector<std::string> b_chrom;
    std::vector<int64_t> b_start, b_end;
    std::vector<int64_t> b_strand;
    std::vector<std::string> b_chromsrc;
    std::vector<int64_t> b_startsrc, b_endsrc;
    std::vector<int64_t> b_strandsrc;
    std::vector<int64_t> b_chain_id;
    std::vector<double> b_score;

    std::unordered_map<std::string, int64_t> src_chrom_sizes;

    bool in_chain = false;
    bool skip_chain = false;
    std::string src_chrom;
    int64_t src_size = 0;
    int src_strand = 0;
    int64_t src_start = 0;
    int64_t src_end = 0;
    std::string tgt_chrom;
    int64_t tgt_size = 0;
    int tgt_strand = 0;
    int64_t tgt_start = 0;
    int64_t tgt_end = 0;
    int64_t chain_id = 0;
    double chain_score = 0.0;
    int64_t cur_src_pos = 0;
    int64_t cur_tgt_pos = 0;

    char *line = nullptr;
    size_t cap = 0;
    ssize_t nread = 0;
    long lineno = 0;

    auto cleanup_line = [&] { if (line) { free(line); line = nullptr; } };

    try {
        while ((nread = getline(&line, &cap, fp)) != -1) {
            ++lineno;
            while (nread > 0 && (line[nread - 1] == '\n' || line[nread - 1] == '\r'
                                 || line[nread - 1] == ' ' || line[nread - 1] == '\t')) {
                line[--nread] = '\0';
            }
            char *l = line;
            while (*l == ' ' || *l == '\t') ++l;
            if (*l == '\0') {
                in_chain = false;
                continue;
            }
            if (*l == '#') {
                continue;
            }

            char *toks[16];
            size_t ntok = split_ws(l, toks, 16);
            if (ntok == 0) {
                in_chain = false;
                continue;
            }

            if (!strcmp(toks[0], "chain")) {
                if (ntok != 13) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: expected 13 fields in chain "
                        "header, got %zu", path, lineno, ntok);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                char *endp = nullptr;
                chain_score = strtod(toks[1], &endp);
                if (*endp) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid chain score", path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                if (!std::isnan(min_score) && chain_score < min_score) {
                    skip_chain = true;
                    in_chain = true;
                    continue;
                }
                skip_chain = false;

                src_chrom = toks[2];

                src_size = strtoll(toks[3], &endp, 10);
                if (*endp || src_size <= 0) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid source chrom size", path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                {
                    auto it = src_chrom_sizes.find(src_chrom);
                    if (it == src_chrom_sizes.end()) {
                        src_chrom_sizes.emplace(src_chrom, src_size);
                    } else if (it->second != src_size) {
                        PyErr_Format(PyExc_ValueError,
                            "Chain file %s, line %ld: source chrom size (%lld) "
                            "differs from previous (%lld)",
                            path, lineno, (long long)src_size, (long long)it->second);
                        cleanup_line();
                        fclose(fp);
                        return nullptr;
                    }
                }

                if (!strcmp(toks[4], "+"))      src_strand = 0;
                else if (!strcmp(toks[4], "-")) src_strand = 1;
                else {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid source strand '%s'",
                        path, lineno, toks[4]);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                src_start = strtoll(toks[5], &endp, 10);
                if (*endp || src_start < 0 || src_start >= src_size) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: source start out of range",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                src_end = strtoll(toks[6], &endp, 10);
                if (*endp || src_end <= src_start || src_end > src_size) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: source end out of range",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                const char *tgt_chrom_raw = toks[7];
                int tgt_chromid = -1;
                try {
                    tgt_chromid = chromkey.chrom2id(tgt_chrom_raw);
                } catch (TGLException &) {
                    tgt_chromid = -1;
                }

                int64_t tgt_size_decl = strtoll(toks[8], &endp, 10);
                if (*endp || tgt_size_decl <= 0) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid target chrom size",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                if (tgt_chromid < 0) {
                    skip_chain = true;
                    in_chain = true;
                    cur_src_pos = src_start;
                    int64_t tgt_start_raw = strtoll(toks[10], &endp, 10);
                    cur_tgt_pos = tgt_start_raw;
                    continue;
                }

                int64_t db_size = (int64_t)chromkey.get_chrom_size(tgt_chromid);
                if (tgt_size_decl != db_size) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: target chrom size (%lld) "
                        "differs from database (%lld)",
                        path, lineno, (long long)tgt_size_decl, (long long)db_size);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                tgt_chrom = chromkey.id2chrom(tgt_chromid);
                tgt_size = tgt_size_decl;

                if (!strcmp(toks[9], "+"))      tgt_strand = 0;
                else if (!strcmp(toks[9], "-")) tgt_strand = 1;
                else {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid target strand '%s'",
                        path, lineno, toks[9]);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                tgt_start = strtoll(toks[10], &endp, 10);
                if (*endp || tgt_start < 0 || tgt_start >= tgt_size) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: target start out of range",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                tgt_end = strtoll(toks[11], &endp, 10);
                if (*endp || tgt_end <= tgt_start || tgt_end > tgt_size) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: target end out of range",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                chain_id = strtoll(toks[12], &endp, 10);
                if (*endp) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: invalid chain id",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }

                cur_src_pos = src_start;
                cur_tgt_pos = tgt_start;
                in_chain = true;
                continue;
            }

            if (!in_chain) {
                PyErr_Format(PyExc_ValueError,
                    "Chain file %s, line %ld: alignment block outside chain",
                    path, lineno);
                cleanup_line();
                fclose(fp);
                return nullptr;
            }
            if (ntok != 1 && ntok != 3) {
                PyErr_Format(PyExc_ValueError,
                    "Chain file %s, line %ld: expected 1 or 3 fields in block "
                    "line, got %zu", path, lineno, ntok);
                cleanup_line();
                fclose(fp);
                return nullptr;
            }

            char *endp = nullptr;
            int64_t size = strtoll(toks[0], &endp, 10);
            if (*endp || size <= 0) {
                PyErr_Format(PyExc_ValueError,
                    "Chain file %s, line %ld: invalid block size", path, lineno);
                cleanup_line();
                fclose(fp);
                return nullptr;
            }

            if (skip_chain) {
                continue;
            }

            int64_t block_src_start, block_src_end;
            if (src_strand == 0) {
                block_src_start = cur_src_pos;
                block_src_end = cur_src_pos + size;
            } else {
                block_src_start = src_size - cur_src_pos - size;
                block_src_end = src_size - cur_src_pos;
            }

            int64_t block_tgt_start, block_tgt_end;
            if (tgt_strand == 0) {
                block_tgt_start = cur_tgt_pos;
                block_tgt_end = cur_tgt_pos + size;
            } else {
                block_tgt_start = tgt_size - cur_tgt_pos - size;
                block_tgt_end = tgt_size - cur_tgt_pos;
            }

            b_chrom.push_back(tgt_chrom);
            b_start.push_back(block_tgt_start);
            b_end.push_back(block_tgt_end);
            b_strand.push_back(tgt_strand);
            b_chromsrc.push_back(src_chrom);
            b_startsrc.push_back(block_src_start);
            b_endsrc.push_back(block_src_end);
            b_strandsrc.push_back(src_strand);
            b_chain_id.push_back(chain_id);
            b_score.push_back(chain_score);

            if (ntok == 3) {
                int64_t dt = strtoll(toks[1], &endp, 10);
                if (*endp || dt < 0) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: negative or invalid dt",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                int64_t dq = strtoll(toks[2], &endp, 10);
                if (*endp || dq < 0) {
                    PyErr_Format(PyExc_ValueError,
                        "Chain file %s, line %ld: negative or invalid dq",
                        path, lineno);
                    cleanup_line();
                    fclose(fp);
                    return nullptr;
                }
                cur_src_pos += size + dt;
                cur_tgt_pos += size + dq;
            } else {
                cur_src_pos += size;
                cur_tgt_pos += size;
            }
        }
        if (ferror(fp)) {
            PyErr_Format(PyExc_IOError, "Error reading %s: %s", path, strerror(errno));
            cleanup_line();
            fclose(fp);
            return nullptr;
        }
    } catch (...) {
        cleanup_line();
        fclose(fp);
        throw;
    }
    cleanup_line();
    fclose(fp);

    if (b_chrom.empty()) {
        return_none();
    }

    npy_intp n_out = (npy_intp)b_chrom.size();
    PMPY py_chrom(PyArray_SimpleNew(1, &n_out, NPY_OBJECT), true);
    PMPY py_chromsrc(PyArray_SimpleNew(1, &n_out, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_strand(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_startsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_endsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_strandsrc(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_chain_id(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_score(PyArray_SimpleNew(1, &n_out, NPY_DOUBLE), true);
    if (!py_chrom || !py_chromsrc || !py_start || !py_end || !py_strand
        || !py_startsrc || !py_endsrc || !py_strandsrc
        || !py_chain_id || !py_score) {
        return_err();
    }

    PyObject **chrom_out    = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    PyObject **chromsrc_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chromsrc);
    int64_t *start_out      = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out        = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    int64_t *strand_out     = (int64_t *)PyArray_DATA((PyArrayObject *)*py_strand);
    int64_t *startsrc_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_startsrc);
    int64_t *endsrc_out     = (int64_t *)PyArray_DATA((PyArrayObject *)*py_endsrc);
    int64_t *strandsrc_out  = (int64_t *)PyArray_DATA((PyArrayObject *)*py_strandsrc);
    int64_t *chain_id_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_chain_id);
    double *score_out       = (double *)PyArray_DATA((PyArrayObject *)*py_score);

    for (npy_intp i = 0; i < n_out; ++i) {
        chrom_out[i] = nullptr;
        chromsrc_out[i] = nullptr;
    }
    for (npy_intp i = 0; i < n_out; ++i) {
        PyObject *s1 = PyUnicode_FromStringAndSize(b_chrom[i].data(), b_chrom[i].size());
        if (!s1) {
            for (npy_intp j = i; j < n_out; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
                Py_INCREF(Py_None); chromsrc_out[j] = Py_None;
            }
            return_err();
        }
        chrom_out[i] = s1;
        PyObject *s2 = PyUnicode_FromStringAndSize(b_chromsrc[i].data(), b_chromsrc[i].size());
        if (!s2) {
            for (npy_intp j = i + 1; j < n_out; ++j) {
                Py_INCREF(Py_None); chrom_out[j] = Py_None;
            }
            for (npy_intp j = i; j < n_out; ++j) {
                Py_INCREF(Py_None); chromsrc_out[j] = Py_None;
            }
            return_err();
        }
        chromsrc_out[i] = s2;
        start_out[i]     = b_start[i];
        end_out[i]       = b_end[i];
        strand_out[i]    = b_strand[i];
        startsrc_out[i]  = b_startsrc[i];
        endsrc_out[i]    = b_endsrc[i];
        strandsrc_out[i] = b_strandsrc[i];
        chain_id_out[i]  = b_chain_id[i];
        score_out[i]     = b_score[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return_err();
    PyDict_SetItemString(result, "chrom",     py_chrom);
    PyDict_SetItemString(result, "start",     py_start);
    PyDict_SetItemString(result, "end",       py_end);
    PyDict_SetItemString(result, "strand",    py_strand);
    PyDict_SetItemString(result, "chromsrc",  py_chromsrc);
    PyDict_SetItemString(result, "startsrc",  py_startsrc);
    PyDict_SetItemString(result, "endsrc",    py_endsrc);
    PyDict_SetItemString(result, "strandsrc", py_strandsrc);
    PyDict_SetItemString(result, "chain_id",  py_chain_id);
    PyDict_SetItemString(result, "score",     py_score);

    return_py(result);

    } catch (TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return_err();
    }
}
