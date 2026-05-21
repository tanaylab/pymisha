#ifndef PMCHAININTERVALS_H
#define PMCHAININTERVALS_H

#include <cstdint>
#include <string>
#include <vector>

// POD view of source intervals to map.
// All arrays must have length n; value may be nullptr (no value column).
// Strings in *chrom are owned by the caller; the helper does not copy them
// (it interns into a local lookup keyed by string content).
struct SrcRowInput {
    const std::string *chrom;   // length n
    const std::int64_t *start;  // length n
    const std::int64_t *end;    // length n
    const double *value;        // length n or nullptr (no value column)
    std::int64_t n;
};

// POD view of one chain. All arrays must have length n. Strings owned by caller.
struct ChainRowInput {
    const std::string *chrom;       // length n: target chrom names
    const std::int64_t *start;      // length n: tgt start
    const std::int64_t *end;        // length n: tgt end
    const std::int64_t *strand;     // length n: tgt strand (0=+, 1=-)
    const std::string *chromsrc;    // length n: src chrom names
    const std::int64_t *startsrc;   // length n: src start
    const std::int64_t *endsrc;     // length n: src end
    const std::int64_t *strandsrc;  // length n: src strand
    const std::int64_t *chain_id;   // length n
    const double *score;            // length n
    std::int64_t n;
};

// Output buffers (helper APPENDS - caller should pre-clear).
// chrom is materialized as std::vector<std::string> (helper resolves the
// internal interner and writes back the strings).
struct MappedOutput {
    std::vector<std::string> chrom;
    std::vector<std::int64_t> start;
    std::vector<std::int64_t> end;
    std::vector<std::int64_t> intervalID;
    std::vector<std::int64_t> chain_id;
    // Source-space helper columns: common_start/common_end of the chain ∩ src
    // overlap. Emitted as __src_start/__src_end in the Python dict. Used by
    // canonic_merge in the SPARSE pipeline; the FIXED_BIN orchestrator (Task 6)
    // does not consume them.
    std::vector<std::int64_t> src_start;
    std::vector<std::int64_t> src_end;
    std::vector<double>       score;       // populated only if include_metadata
    std::vector<double>       value;       // populated only if src.value != nullptr
};

// Cluster strategy. NONE / "" means no cluster resolution.
enum class ClusterStrat { NONE, UNION, SUM, MAX };

// Parse cluster strategy string. Throws std::invalid_argument on unknown value.
// "" or "none" -> NONE.
ClusterStrat parse_cluster_strategy_str(const std::string &s);

// Map src intervals through chain. include_metadata controls whether out.score
// gets populated. The helper sorts a local copy of the chain by
// (chromid_src, start_src, end_src). Throws std::invalid_argument on bad
// input (e.g. negative start/end).
void map_intervals_cpp(
    const SrcRowInput &src,
    const ChainRowInput &chain,
    bool include_metadata,
    ClusterStrat cluster_strategy,
    MappedOutput &out);

#endif  // PMCHAININTERVALS_H
