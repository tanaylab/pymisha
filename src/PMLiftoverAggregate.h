#ifndef PMLIFTOVERAGGREGATE_H
#define PMLIFTOVERAGGREGATE_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

enum class AggType {
    MEAN, MEDIAN, SUM, MIN, MAX, COUNT, FIRST, LAST, NTH,
    MAX_COV_LEN, MIN_COV_LEN, MAX_COV_FRAC, MIN_COV_FRAC,
};

// Parse agg type string ("mean", ..., "nth", "max.coverage_len", "min.coverage_len",
// "max.coverage_frac", "min.coverage_frac"). Throws std::invalid_argument on unknown value.
AggType parse_agg_type_str(const std::string &s);

// One contribution to a target locus, as R's AggregationHelpers.h Contribution:
// overlap_len is the part of the locus it covers and coverage_frac that length over
// the locus length. start / end order first / last / nth; chain_id merges the pieces
// of one contribution.
struct BinContribution {
    double value;
    double overlap_len;
    double coverage_frac;
    int64_t start;
    int64_t end;
    bool is_na;
    int64_t chain_id;
};

// The value of a target locus from its contributions: R's aggregate_values
// (AggregationHelpers.h). min_n_or_negative < 0 means no minimum; nth_index is
// 1-based and used by NTH only.
double aggregate_value_for_bin(
    const std::vector<BinContribution> &state,
    AggType agg, bool na_rm,
    int64_t min_n_or_negative, int64_t nth_index);

// SPARSE-path aggregator, as R's gtrack_liftover writes a sparse track
// (GTrackLiftover.cpp): per chrom (in name order), the intervals sorted by
// (start, end, key, value), every run of overlapping intervals becomes one
// interval whose value is R's aggregate_values over the run - an interval
// counting its whole length, intervals sharing a key (the pieces of one source
// interval through one chain) counting once. A NaN value is kept, so a run of
// NaN source values gives a NaN interval. Touching intervals are not merged.
// Inputs need not be sorted.
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
    const std::vector<std::int64_t> &in_key,
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
// ceil(chrom_size / bin_size) rows per target chrom. A bin with no
// contributions gets what R's aggregate_values gives it (NaN, or 0 for
// "count" without min_n); a chrom with none at all is NaN throughout, as R
// writes it.
//
// Matches R GTrackLiftover.cpp:702-768 + AggregationHelpers.h::aggregate_values.
//
// in_*: per-interval contributions (NOT sorted; helper sorts internally
//       per target chrom, in R's order: start, end, chain id, value).
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
