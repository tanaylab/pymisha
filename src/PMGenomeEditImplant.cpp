/*
 * PMGenomeEditImplant.cpp
 *
 * C++ fast path for ggenome_implant: read a reference FASTA, apply
 * perturbations (donor sequence replacements), write output FASTA + .fai.
 */

#include "pymisha.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Perturbation {
    int start;         // 0-based
    int end;           // 0-based, exclusive
    const char *donor; // pointer into Python string storage (valid for call duration)
};

struct FaiEntry {
    string name;
    long long length;
    long long offset;
    int linebases;
    int linewidth;
};

} // anonymous namespace

/*
 * pm_ggenome_implant(genome_fasta, output, chroms, starts, ends, donors, line_width)
 *
 * genome_fasta : str — path to reference FASTA
 * output       : str — path for output FASTA
 * chroms       : list[str] — perturbation chromosomes
 * starts       : list[int] — perturbation starts (0-based)
 * ends         : list[int] — perturbation ends (0-based, exclusive)
 * donors       : list[str] — donor sequences (uppercased by caller)
 * line_width   : int — bases per FASTA line
 *
 * Returns: list of tuples [(name, length, offset, linebases, linewidth), ...]
 */
PyObject *pm_ggenome_implant(PyObject *self, PyObject *args)
{
    const char *genome_fasta = nullptr;
    const char *output_path = nullptr;
    PyObject *py_chroms = nullptr;
    PyObject *py_starts = nullptr;
    PyObject *py_ends = nullptr;
    PyObject *py_donors = nullptr;
    int line_width = 80;

    if (!PyArg_ParseTuple(args, "ssOOOOi",
                          &genome_fasta, &output_path,
                          &py_chroms, &py_starts, &py_ends,
                          &py_donors, &line_width)) {
        return nullptr;
    }

    try {
        // --- validate list args ---
        if (!PyList_Check(py_chroms) || !PyList_Check(py_starts) ||
            !PyList_Check(py_ends) || !PyList_Check(py_donors)) {
            PyErr_SetString(PyExc_TypeError,
                "chroms, starts, ends, donors must be lists");
            return nullptr;
        }

        Py_ssize_t n_perts = PyList_Size(py_chroms);
        if (PyList_Size(py_starts) != n_perts ||
            PyList_Size(py_ends) != n_perts ||
            PyList_Size(py_donors) != n_perts) {
            PyErr_SetString(PyExc_ValueError,
                "chroms, starts, ends, donors must have the same length");
            return nullptr;
        }

        // --- build perturbation index ---
        map<string, vector<Perturbation>> pert_map;
        for (Py_ssize_t i = 0; i < n_perts; i++) {
            Perturbation p;
            p.start = (int)PyLong_AsLong(PyList_GetItem(py_starts, i));
            p.end = (int)PyLong_AsLong(PyList_GetItem(py_ends, i));
            p.donor = PyUnicode_AsUTF8(PyList_GetItem(py_donors, i));
            if (PyErr_Occurred()) return nullptr;

            const char *chrom = PyUnicode_AsUTF8(PyList_GetItem(py_chroms, i));
            if (!chrom) return nullptr;
            pert_map[string(chrom)].push_back(p);
        }

        // Sort descending by start
        for (auto &kv : pert_map) {
            sort(kv.second.begin(), kv.second.end(),
                 [](const Perturbation &a, const Perturbation &b) {
                     return a.start > b.start;
                 });
        }

        // --- open files ---
        FILE *fin = fopen(genome_fasta, "r");
        if (!fin) {
            PyErr_Format(PyExc_FileNotFoundError,
                "Cannot open input FASTA: %s", genome_fasta);
            return nullptr;
        }

        FILE *fout = fopen(output_path, "w");
        if (!fout) {
            fclose(fin);
            PyErr_Format(PyExc_IOError,
                "Cannot open output file: %s", output_path);
            return nullptr;
        }

        // 4MB I/O buffers
        const size_t BUF_SIZE = 4 * 1024 * 1024;
        vector<char> inbuf(BUF_SIZE);
        vector<char> outbuf(BUF_SIZE);
        setvbuf(fin, inbuf.data(), _IOFBF, BUF_SIZE);
        setvbuf(fout, outbuf.data(), _IOFBF, BUF_SIZE);

        // --- streaming state ---
        vector<char> seq;
        seq.reserve(256 * 1024 * 1024);
        string current_chrom;
        vector<FaiEntry> fai_entries;
        long long byte_offset = 0;
        bool had_error = false;
        string error_msg;

        const size_t LINE_BUF = 65536;
        vector<char> line(LINE_BUF);

        // --- flush one chromosome ---
        auto flush_chrom = [&]() {
            if (current_chrom.empty() || had_error) return;

            long long seq_len = (long long)seq.size();

            auto it = pert_map.find(current_chrom);
            if (it != pert_map.end()) {
                for (const auto &p : it->second) {
                    if (p.start < 0 || p.end > seq_len) {
                        had_error = true;
                        char buf[512];
                        snprintf(buf, sizeof(buf),
                            "Interval %s:%d-%d is out of bounds "
                            "(chrom length: %lld)",
                            current_chrom.c_str(), p.start, p.end, seq_len);
                        error_msg = buf;
                        return;
                    }
                    int donor_len = p.end - p.start;
                    memcpy(&seq[p.start], p.donor, donor_len);
                }
            }

            fprintf(fout, ">%s\n", current_chrom.c_str());
            long long header_bytes = (long long)current_chrom.size() + 2;

            long long pos = 0;
            while (pos < seq_len) {
                long long chunk = min((long long)line_width, seq_len - pos);
                fwrite(&seq[pos], 1, chunk, fout);
                fputc('\n', fout);
                pos += chunk;
            }

            FaiEntry entry;
            entry.name = current_chrom;
            entry.length = seq_len;
            entry.offset = byte_offset + header_bytes;
            entry.linebases = (seq_len > 0) ? (int)min((long long)line_width, seq_len) : 0;
            entry.linewidth = (seq_len > 0) ? entry.linebases + 1 : 0;
            fai_entries.push_back(entry);

            long long n_full_lines = seq_len / line_width;
            long long remainder = seq_len % line_width;
            long long data_bytes = n_full_lines * (line_width + 1);
            if (remainder > 0) data_bytes += remainder + 1;
            byte_offset += header_bytes + data_bytes;

            seq.clear();
        };

        // --- read input FASTA ---
        while (fgets(line.data(), LINE_BUF, fin)) {
            if (line[0] == '>') {
                flush_chrom();
                if (had_error) break;

                char *name_start = line.data() + 1;
                char *p = name_start;
                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
                current_chrom.assign(name_start, p - name_start);
            } else {
                for (char *p = line.data(); *p && *p != '\n' && *p != '\r'; p++) {
                    seq.push_back((char)toupper((unsigned char)*p));
                }
            }
        }
        if (!had_error) flush_chrom();

        fclose(fin);

        if (had_error) {
            fclose(fout);
            remove(output_path);
            PyErr_SetString(PyExc_ValueError, error_msg.c_str());
            return nullptr;
        }

        // --- write .fai ---
        string fai_path = string(output_path) + ".fai";
        FILE *ffai = fopen(fai_path.c_str(), "w");
        if (!ffai) {
            fclose(fout);
            PyErr_Format(PyExc_IOError,
                "Cannot open .fai file: %s", fai_path.c_str());
            return nullptr;
        }
        for (const auto &e : fai_entries) {
            fprintf(ffai, "%s\t%lld\t%lld\t%d\t%d\n",
                    e.name.c_str(), e.length, e.offset,
                    e.linebases, e.linewidth);
        }
        fclose(ffai);
        fclose(fout);

        // --- return FAI as list of tuples ---
        Py_ssize_t n = (Py_ssize_t)fai_entries.size();
        PyObject *result = PyList_New(n);
        if (!result) return nullptr;

        for (Py_ssize_t i = 0; i < n; i++) {
            const auto &e = fai_entries[i];
            PyObject *tup = Py_BuildValue("(sLLii)",
                e.name.c_str(), e.length, e.offset,
                e.linebases, e.linewidth);
            if (!tup) {
                Py_DECREF(result);
                return nullptr;
            }
            PyList_SET_ITEM(result, i, tup);
        }

        return result;

    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
