#ifndef PMSOURCETRACK_H
#define PMSOURCETRACK_H

#include <cstdint>
#include <string>
#include <vector>

// Internal C++ helper called by both pm_read_source_track_1d (Python entry
// point) and pm_liftover_track (G1.P3.C orchestrator). Reads a source-track
// directory and fills the provided vectors with intervals + values.
//
// On success:
//   - out_type is "dense" or "sparse" (sparse for empty directories).
//   - out_bin_size is the bin size in bp for dense sources (R-parity check:
//     all dense per-chrom files must report the SAME bin_size; mismatch
//     throws std::invalid_argument matching R's error message format).
//     For sparse / empty sources, out_bin_size is 0.
//   - rows are APPENDED to the four output vectors (caller should pre-clear).
//
// Throws std::invalid_argument / std::runtime_error on I/O or format errors
// (including bin_size mismatch across dense per-chrom files). The Python
// wrapper translates these into Python exceptions.
//
// NOTE: this helper does NOT set the Python error state directly. All internal
// errors throw std::exception subclasses (std::invalid_argument for data
// format errors, std::runtime_error for I/O errors). The Python wrapper
// translates these into the appropriate Python exceptions.
void read_source_track_1d_cpp(
    const std::string &src_track_dir,
    std::string &out_type,
    std::int64_t &out_bin_size,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value);

#endif  // PMSOURCETRACK_H
