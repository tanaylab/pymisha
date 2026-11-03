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
//   - rows are APPENDED to the four output vectors (caller should pre-clear),
//     one per dense bin or sparse interval, NaN values included and an
//     infinite value read as NaN, as R's GenomeTrackFixedBin /
//     GenomeTrackSparse read them.
//
// Throws std::invalid_argument / std::runtime_error on I/O or format errors
// (including bin_size mismatch across dense per-chrom files). The Python
// wrapper translates these into Python exceptions.
//
// NOTE: this helper does NOT set the Python error state directly. All internal
// errors throw std::exception subclasses (std::invalid_argument for data
// format errors, std::runtime_error for I/O errors). The Python wrapper
// translates these into the appropriate Python exceptions.
//
// src_chromkey: the chain's source chromosomes, with the "chr"-toggled names as their
// aliases (R's gtrack.liftover): the per-chromosome files are then the ones every reader
// finds for these chromosomes (GenomeTrack::ChromFiles), a file named by a chromosome or
// an alias of it, and their rows are named by the chain's names. NULL: every file, by its
// own name.
// db_chroms: the chromosome names by chrom id of the database that holds an indexed source
// (R's .gdb.chrom_names_at); NULL: load_source_chrom_names.
class GenomeChromKey;
void read_source_track_1d_cpp(
    const std::string &src_track_dir,
    std::string &out_type,
    std::int64_t &out_bin_size,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value,
    const GenomeChromKey *src_chromkey = nullptr,
    const std::vector<std::string> *db_chroms = nullptr);

// The chain's source chromosomes in a key (no sizes), with the names with the "chr" prefix
// toggled as their aliases: a per-chromosome source track may name its files either way (as
// earlier pymisha did). R's gtrack_liftover builds the same key.
void chain_src_chromkey(const std::vector<std::string> &src_names, GenomeChromKey &key);

// The database that holds the track directory src_track_dir: the parent of the
// nearest enclosing "tracks" directory that has a chrom_sizes.txt next to it,
// found from the resolved path. Returns false if no database holds the track.
bool source_db_root_from_track_dir(const std::string &src_track_dir, std::string &db_root_out);

// The chromosome names of the database at db_root by chrom id, as PMDb gives
// them (gdb_chrom_order). An indexed track (track.idx) is keyed by these ids.
// Throws std::invalid_argument on failure.
void load_source_chrom_names(const std::string &db_root, std::vector<std::string> &names_out);

#endif  // PMSOURCETRACK_H
