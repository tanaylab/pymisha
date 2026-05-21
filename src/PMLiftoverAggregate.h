#ifndef PMLIFTOVERAGGREGATE_H
#define PMLIFTOVERAGGREGATE_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

enum class AggType {
    MEAN, MEDIAN, SUM, MIN, MAX, COUNT, FIRST, LAST, NTH,
};

// Parse agg type string. Throws std::invalid_argument on unknown value.
AggType parse_agg_type_str(const std::string &s);

// SPARSE-path aggregator: aggregate overlapping intervals by chrom using
// breakpoint-sweep, producing one output row per maximal-constant segment.
// Inputs need not be sorted (helper sorts internally).
//
// APPENDS to output vectors (caller should pre-clear).
// min_n_or_negative: minimum non-NaN count required; < 0 means no minimum.
// nth_index: 1-based index for NTH aggregation; ignored for other types.
//
// Throws std::invalid_argument on malformed input (length mismatches, etc.).
void aggregate_overlapping_cpp(
    const std::vector<std::string> &in_chrom,
    const std::vector<std::int64_t> &in_start,
    const std::vector<std::int64_t> &in_end,
    const std::vector<double> &in_value,
    AggType agg,
    bool na_rm,
    std::int64_t min_n_or_negative,
    std::int64_t nth_index,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value);

// FIXED_BIN-path aggregator: per output bin per target chrom, collect
// contributions (interval, value, overlap_len, chain_id), merge entries
// sharing the same chain_id (sum overlap_len), apply agg function. Emits
// ceil(chrom_size / bin_size) rows per target chrom - bins with no
// contributions get NaN value (matching R's behavior; the Python wrapper
// will write these out as NaN bins).
//
// Matches R GTrackLiftover.cpp:702-768 + AggregationHelpers.h::aggregate_values.
//
// in_*: per-interval contributions (NOT sorted; helper sorts internally
//       per target chrom).
// tgt_chrom_sizes: ordered list of (chrom_name, chrom_size_bp) pairs for the
//                  target genome. Output row order follows this sequence.
//                  The caller (Python wrapper) iterates the Python dict in
//                  insertion order and builds the vector accordingly.
// bin_size: positive bp width of each output bin. <= 0 throws.
//
// APPENDS to output vectors (caller should pre-clear).
// Throws std::invalid_argument on malformed input.
void aggregate_per_bin_cpp(
    const std::vector<std::string> &in_chrom,
    const std::vector<std::int64_t> &in_start,
    const std::vector<std::int64_t> &in_end,
    const std::vector<double> &in_value,
    const std::vector<std::int64_t> &in_chain_id,
    const std::vector<std::pair<std::string, std::int64_t>> &tgt_chrom_sizes,
    std::int64_t bin_size,
    AggType agg,
    bool na_rm,
    std::int64_t min_n_or_negative,
    std::int64_t nth_index,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value);

#endif  // PMLIFTOVERAGGREGATE_H
