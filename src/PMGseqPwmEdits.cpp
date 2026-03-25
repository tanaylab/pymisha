/*
 * PMGseqPwmEdits.cpp
 *
 * Python C extension: pm_gseq_pwm_edits()
 *
 * Port of R misha's C_gseq_pwm_edits — returns detailed edit information
 * for PWM edit distance. For each input sequence, finds the optimal window
 * and the specific base changes needed to reach the threshold.
 */

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <cctype>
#include <cfloat>

#include "pymisha.h"
#include "DnaPSSM.h"

extern PyObject *s_pm_err;

namespace {

constexpr float kLogZeroThreshold = std::numeric_limits<float>::lowest() * 0.5f;

struct EditInfo {
    int motif_col;    // 1-based motif column (0 for deletions - will be None in Python)
    char ref_base;    // current/deleted base ('\0' for insertions)
    char alt_base;    // replacement/inserted base ('\0' for deletions)
    float gain;       // score improvement from this edit
    std::string edit_type; // "sub", "ins", "del"
};

struct WindowResult {
    int seq_idx;           // 0-based index into input sequences
    int strand;            // +1 or -1
    int window_start;      // 1-based position within sequence
    float score_before;
    float score_after;
    int n_edits;           // total edits needed (0 if already above threshold, -1 if unreachable)
    std::vector<EditInfo> edits;
    std::string window_seq;   // motif-length sequence at the window (as seen by PSSM)
    std::string mutated_seq;  // window_seq with edits applied
};

inline int base_to_index(char base) {
    switch (base) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 4;
    }
}

inline char index_to_base(int idx) {
    static const char bases[] = {'A', 'C', 'G', 'T'};
    return (idx >= 0 && idx < 4) ? bases[idx] : 'N';
}

inline char complement_base(char base) {
    switch (base) {
        case 'A': case 'a': return 'T';
        case 'C': case 'c': return 'G';
        case 'G': case 'g': return 'C';
        case 'T': case 't': return 'A';
        default: return 'N';
    }
}

WindowResult compute_window_edits_detailed(
    const char* seq_ptr, int L,
    const DnaPSSM& pssm,
    const std::vector<float>& col_max_scores,
    float S_max,
    float threshold,
    int max_edits,
    bool reverse,
    const std::vector<float>& gain_values,
    const std::vector<std::vector<uint8_t>>& bin_index)
{
    WindowResult result;
    result.n_edits = -1;
    result.score_before = 0.0f;
    result.score_after = 0.0f;

    struct PosInfo {
        int motif_col;
        char ref_base;
        int ref_idx;
        float base_score;
        float gain;
        int best_base_idx;
        bool mandatory;
    };

    std::vector<PosInfo> positions(L);
    int mandatory_edits = 0;
    double true_score = 0.0;
    double adjusted_score = 0.0;

    for (int i = 0; i < L; i++) {
        PosInfo& p = positions[i];
        p.motif_col = i;

        int seq_idx = reverse ? (L - 1 - i) : i;
        char base = seq_ptr[seq_idx];
        if (reverse) {
            base = complement_base(base);
        }
        p.ref_base = base;
        p.ref_idx = base_to_index(base);

        float best_score = -std::numeric_limits<float>::infinity();
        p.best_base_idx = 0;
        for (int b = 0; b < 4; b++) {
            float s = pssm[i].get_log_prob_from_code(b);
            if (s > best_score) {
                best_score = s;
                p.best_base_idx = b;
            }
        }

        if (p.ref_idx == 4) {
            float mean_logp = 0.0f;
            for (int b = 0; b < 4; b++) mean_logp += pssm[i].get_log_prob_from_code(b);
            mean_logp /= 4.0f;

            p.mandatory = true;
            p.base_score = mean_logp;
            p.gain = col_max_scores[i] - mean_logp;
            mandatory_edits++;
            true_score += static_cast<double>(mean_logp);
            adjusted_score += static_cast<double>(col_max_scores[i]);
        } else {
            float score = pssm[i].get_log_prob_from_code(p.ref_idx);
            if (score <= kLogZeroThreshold || !std::isfinite(score)) {
                p.mandatory = true;
                p.base_score = score;
                p.gain = col_max_scores[i] - score;
                mandatory_edits++;
                true_score += static_cast<double>(score);
                adjusted_score += static_cast<double>(col_max_scores[i]);
            } else {
                p.mandatory = false;
                p.base_score = score;
                p.gain = col_max_scores[i] - score;
                true_score += static_cast<double>(score);
                adjusted_score += static_cast<double>(score);
            }
        }
    }

    result.score_before = static_cast<float>(true_score);

    result.window_seq.resize(L);
    for (int i = 0; i < L; i++) {
        result.window_seq[i] = positions[i].ref_base;
    }

    double deficit = static_cast<double>(threshold) - adjusted_score;

    if (deficit <= 0.0) {
        result.n_edits = mandatory_edits;
        result.score_after = static_cast<float>(adjusted_score);
        result.mutated_seq = result.window_seq;
        for (int i = 0; i < L; i++) {
            if (positions[i].mandatory) {
                EditInfo edit;
                edit.motif_col = i + 1;
                edit.ref_base = positions[i].ref_base;
                edit.alt_base = index_to_base(positions[i].best_base_idx);
                edit.gain = positions[i].gain;
                edit.edit_type = "sub";
                result.edits.push_back(edit);
                result.mutated_seq[i] = edit.alt_base;
            }
        }
        return result;
    }

    double max_possible_gain = static_cast<double>(S_max) - adjusted_score;
    if (max_possible_gain < deficit) {
        result.n_edits = -1;
        return result;
    }

    std::vector<int> sorted_positions;
    sorted_positions.reserve(L);
    for (int i = 0; i < L; i++) {
        if (!positions[i].mandatory && positions[i].gain > 0.0f) {
            sorted_positions.push_back(i);
        }
    }
    std::sort(sorted_positions.begin(), sorted_positions.end(),
              [&positions](int a, int b) {
                  return positions[a].gain > positions[b].gain;
              });

    double acc = 0.0;
    int edits = mandatory_edits;
    std::vector<EditInfo> edit_list;

    for (int i = 0; i < L; i++) {
        if (positions[i].mandatory) {
            EditInfo edit;
            edit.motif_col = i + 1;
            edit.ref_base = positions[i].ref_base;
            edit.alt_base = index_to_base(positions[i].best_base_idx);
            edit.gain = positions[i].gain;
            edit.edit_type = "sub";
            edit_list.push_back(edit);
        }
    }

    for (int idx : sorted_positions) {
        if (max_edits >= 0 && edits >= max_edits) {
            break;
        }

        acc += static_cast<double>(positions[idx].gain);
        edits++;

        EditInfo edit;
        edit.motif_col = idx + 1;
        edit.ref_base = positions[idx].ref_base;
        edit.alt_base = index_to_base(positions[idx].best_base_idx);
        edit.gain = positions[idx].gain;
        edit.edit_type = "sub";
        edit_list.push_back(edit);

        if (acc >= deficit) {
            result.n_edits = edits;
            result.score_after = static_cast<float>(adjusted_score + acc);
            result.edits = edit_list;
            result.mutated_seq = result.window_seq;
            for (const auto& e : result.edits) {
                result.mutated_seq[e.motif_col - 1] = e.alt_base;
            }
            return result;
        }
    }

    result.n_edits = -1;
    return result;
}

/**
 * Compute detailed edits for a single window with indel support.
 *
 * Uses banded Needleman-Wunsch DP (same algorithm as
 * PWMEditDistanceScorer::compute_with_indels) but additionally tracks
 * per-edit info (type, position, base, gain).
 *
 * For each window length W in [L-D, L+D], aligns motif[0..L-1] against
 * seq[0..W-1] using a 3D DP table dp[i][j][k], then traces back the
 * alignment to identify specific edits.
 */
WindowResult compute_window_edits_detailed_with_indels(
    const char* seq_ptr, int seq_avail, int L,
    const DnaPSSM& pssm,
    const std::vector<float>& col_max_scores,
    float S_max,
    float threshold,
    int max_edits,
    int max_indels,
    bool reverse)
{
    const int D = max_indels;

    WindowResult best_result;
    best_result.n_edits = -1;
    best_result.score_before = 0.0f;
    best_result.score_after = 0.0f;

    // Try each window length W in [L-D, L+D]
    for (int W = std::max(1, L - D); W <= L + D; ++W) {
        if (W > seq_avail) break;

        const int rows = L + 1;
        const int cols = W + 1;
        const int indel_levels = D + 1;

        // Flattened 3D DP table
        std::vector<double> dp(rows * cols * indel_levels,
                               -std::numeric_limits<double>::infinity());

        auto idx3 = [cols, indel_levels](int i, int j, int k) -> int {
            return i * cols * indel_levels + j * indel_levels + k;
        };

        // Base case
        dp[idx3(0, 0, 0)] = 0.0;

        // First column: skip motif positions (insertions - each costs 1 indel)
        for (int i = 1; i <= std::min(L, D); ++i) {
            dp[idx3(i, 0, i)] = 0.0;
        }

        // First row: skip sequence positions (deletions - each costs 1 indel)
        for (int j = 1; j <= std::min(W, D); ++j) {
            dp[idx3(0, j, j)] = 0.0;
        }

        // Fill DP
        for (int i = 1; i <= L; ++i) {
            int j_min = std::max(1, i - D);
            int j_max = std::min(W, i + D);

            for (int j = j_min; j <= j_max; ++j) {
                // Get sequence base at position j-1
                int seq_idx = reverse ? (W - 1 - (j - 1)) : (j - 1);
                char base = seq_ptr[seq_idx];
                if (reverse) base = complement_base(base);
                int bidx = base_to_index(base);

                float base_score;
                if (bidx == 4) {
                    float min_s = std::numeric_limits<float>::infinity();
                    for (int b = 0; b < 4; b++) {
                        float s = pssm[i - 1].get_log_prob_from_code(b);
                        if (s < min_s) min_s = s;
                    }
                    base_score = min_s;
                } else {
                    base_score = pssm[i - 1].get_log_prob_from_code(bidx);
                }

                for (int k = 0; k <= D; ++k) {
                    // 1. Match/Substitution (diagonal)
                    if (std::abs((i - 1) - (j - 1)) <= D) {
                        double prev = dp[idx3(i - 1, j - 1, k)];
                        if (prev > -std::numeric_limits<double>::infinity() * 0.5) {
                            double new_score = prev + static_cast<double>(base_score);
                            if (new_score > dp[idx3(i, j, k)]) {
                                dp[idx3(i, j, k)] = new_score;
                            }
                        }
                    }

                    if (k < D) {
                        // 2. Insertion: skip motif[i-1], advance motif not sequence
                        if (std::abs((i - 1) - j) <= D) {
                            double prev = dp[idx3(i - 1, j, k)];
                            if (prev > -std::numeric_limits<double>::infinity() * 0.5) {
                                if (prev > dp[idx3(i, j, k + 1)]) {
                                    dp[idx3(i, j, k + 1)] = prev;
                                }
                            }
                        }

                        // 3. Deletion: skip seq[j-1], advance sequence not motif
                        if (std::abs(i - (j - 1)) <= D) {
                            double prev = dp[idx3(i, j - 1, k)];
                            if (prev > -std::numeric_limits<double>::infinity() * 0.5) {
                                if (prev > dp[idx3(i, j, k + 1)]) {
                                    dp[idx3(i, j, k + 1)] = prev;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract results for each indel count k
        for (int k = 0; k <= D; ++k) {
            double score = dp[idx3(L, W, k)];
            if (score <= -std::numeric_limits<double>::infinity() * 0.5) {
                continue;
            }

            // Traceback to find alignment and compute edits
            struct AlignOp {
                char op;        // 'M', 'I', 'D'
                int motif_pos;  // 0-based motif position (for M and I)
                int seq_pos;    // 0-based seq position (for M and D)
                float base_score;   // score at this aligned position (M only)
                int base_idx;       // base index (M only)
                char base_char;     // base character (M and D)
            };

            std::vector<AlignOp> alignment;
            alignment.reserve(L + D);

            int ti = L, tj = W, tk = k;
            bool traceback_ok = true;

            while (ti > 0 || tj > 0) {
                if (ti == 0 && tj > 0 && tk > 0) {
                    // Must be deletion
                    AlignOp op;
                    op.op = 'D';
                    op.motif_pos = -1;
                    op.seq_pos = tj - 1;
                    int s_idx = reverse ? (W - 1 - (tj - 1)) : (tj - 1);
                    char b = seq_ptr[s_idx];
                    if (reverse) b = complement_base(b);
                    op.base_char = b;
                    op.base_score = 0.0f;
                    op.base_idx = -1;
                    alignment.push_back(op);
                    tj--; tk--;
                    continue;
                }
                if (tj == 0 && ti > 0 && tk > 0) {
                    // Must be insertion
                    AlignOp op;
                    op.op = 'I';
                    op.motif_pos = ti - 1;
                    op.seq_pos = -1;
                    op.base_char = '\0';
                    op.base_score = 0.0f;
                    op.base_idx = -1;
                    alignment.push_back(op);
                    ti--; tk--;
                    continue;
                }
                if (ti == 0 || tj == 0) break;

                double cur = dp[idx3(ti, tj, tk)];

                // Try diagonal (match/substitution) first
                bool found = false;
                if (std::abs((ti - 1) - (tj - 1)) <= D) {
                    int s_idx = reverse ? (W - 1 - (tj - 1)) : (tj - 1);
                    char b = seq_ptr[s_idx];
                    if (reverse) b = complement_base(b);
                    int bi = base_to_index(b);

                    float bs;
                    if (bi == 4) {
                        float min_s = std::numeric_limits<float>::infinity();
                        for (int bb = 0; bb < 4; bb++) {
                            float s = pssm[ti - 1].get_log_prob_from_code(bb);
                            if (s < min_s) min_s = s;
                        }
                        bs = min_s;
                    } else {
                        bs = pssm[ti - 1].get_log_prob_from_code(bi);
                    }

                    double prev = dp[idx3(ti - 1, tj - 1, tk)];
                    if (prev > -std::numeric_limits<double>::infinity() * 0.5 &&
                        std::fabs((prev + static_cast<double>(bs)) - cur) < 1e-9 * std::max(1.0, std::fabs(cur))) {
                        AlignOp op;
                        op.op = 'M';
                        op.motif_pos = ti - 1;
                        op.seq_pos = tj - 1;
                        op.base_char = b;
                        op.base_score = bs;
                        op.base_idx = bi;
                        alignment.push_back(op);
                        ti--; tj--;
                        found = true;
                    }
                }

                if (!found && ti > 0 && tk > 0 && std::abs((ti - 1) - tj) <= D) {
                    // Try insertion (skip motif position)
                    double prev = dp[idx3(ti - 1, tj, tk - 1)];
                    if (prev > -std::numeric_limits<double>::infinity() * 0.5 &&
                        std::fabs(prev - cur) < 1e-9 * std::max(1.0, std::fabs(cur))) {
                        AlignOp op;
                        op.op = 'I';
                        op.motif_pos = ti - 1;
                        op.seq_pos = -1;
                        op.base_char = '\0';
                        op.base_score = 0.0f;
                        op.base_idx = -1;
                        alignment.push_back(op);
                        ti--; tk--;
                        found = true;
                    }
                }

                if (!found && tj > 0 && tk > 0 && std::abs(ti - (tj - 1)) <= D) {
                    // Try deletion (skip seq position)
                    double prev = dp[idx3(ti, tj - 1, tk - 1)];
                    if (prev > -std::numeric_limits<double>::infinity() * 0.5 &&
                        std::fabs(prev - cur) < 1e-9 * std::max(1.0, std::fabs(cur))) {
                        AlignOp op;
                        op.op = 'D';
                        op.motif_pos = -1;
                        op.seq_pos = tj - 1;
                        int s_idx = reverse ? (W - 1 - (tj - 1)) : (tj - 1);
                        char b = seq_ptr[s_idx];
                        if (reverse) b = complement_base(b);
                        op.base_char = b;
                        op.base_score = 0.0f;
                        op.base_idx = -1;
                        alignment.push_back(op);
                        tj--; tk--;
                        found = true;
                    }
                }

                if (!found) {
                    traceback_ok = false;
                    break;
                }
            }

            if (!traceback_ok) continue;

            // Reverse alignment (it was built backwards)
            std::reverse(alignment.begin(), alignment.end());

            // Collect gains from aligned (diagonal/M) positions for greedy sub selection
            struct AlignedPos {
                int align_idx;      // index into alignment vector
                float gain;         // col_max - current score
                int best_base_idx;  // index of best base for this column
                bool mandatory;     // unknown base or zero-prob
            };

            std::vector<AlignedPos> aligned_positions;
            aligned_positions.reserve(L);

            for (size_t a = 0; a < alignment.size(); ++a) {
                if (alignment[a].op != 'M') continue;

                int mc = alignment[a].motif_pos;
                float gain = col_max_scores[mc] - alignment[a].base_score;

                // Find best base for this column
                float best_s = -std::numeric_limits<float>::infinity();
                int best_b = 0;
                for (int b = 0; b < 4; b++) {
                    float s = pssm[mc].get_log_prob_from_code(b);
                    if (s > best_s) { best_s = s; best_b = b; }
                }

                bool mandatory = (alignment[a].base_idx == 4) ||
                    (alignment[a].base_score <= kLogZeroThreshold ||
                     !std::isfinite(alignment[a].base_score));

                AlignedPos ap;
                ap.align_idx = static_cast<int>(a);
                ap.gain = gain;
                ap.best_base_idx = best_b;
                ap.mandatory = mandatory;
                aligned_positions.push_back(ap);
            }

            // Determine substitutions needed
            int mandatory_subs = 0;
            double adjusted = score;
            for (auto& ap : aligned_positions) {
                if (ap.mandatory) {
                    adjusted += static_cast<double>(ap.gain);
                    mandatory_subs++;
                }
            }

            int subs_needed;
            if (adjusted >= static_cast<double>(threshold)) {
                subs_needed = mandatory_subs;
            } else {
                // Need additional greedy subs
                double remaining_deficit = static_cast<double>(threshold) - adjusted;

                // Sort non-mandatory by gain descending
                std::vector<const AlignedPos*> optional;
                optional.reserve(aligned_positions.size());
                for (auto& ap : aligned_positions) {
                    if (!ap.mandatory && ap.gain > 1e-12f) {
                        optional.push_back(&ap);
                    }
                }
                std::sort(optional.begin(), optional.end(),
                          [](const AlignedPos* a, const AlignedPos* b) {
                              return a->gain > b->gain;
                          });

                double acc = 0.0;
                subs_needed = mandatory_subs;
                bool reachable = false;
                for (auto* ap : optional) {
                    if (max_edits >= 0 && (k + subs_needed) >= max_edits) break;
                    acc += static_cast<double>(ap->gain);
                    subs_needed++;
                    if (acc >= remaining_deficit) {
                        reachable = true;
                        break;
                    }
                }
                if (!reachable) {
                    // Check total gain
                    double total_gain = 0.0;
                    for (auto* ap : optional) total_gain += static_cast<double>(ap->gain);
                    total_gain += (adjusted - score); // mandatory gains already counted
                    if (total_gain + (adjusted - score) < static_cast<double>(threshold) - score) {
                        continue; // unreachable
                    }
                    if (!reachable) continue;
                }
            }

            int total_edits = k + subs_needed;
            if (max_edits >= 0 && total_edits > max_edits) continue;

            // Is this better than our current best?
            if (best_result.n_edits >= 0 && total_edits >= best_result.n_edits) {
                continue;
            }

            // Build detailed edit list and result
            std::set<int> sub_align_indices;
            // First, mandatory
            for (auto& ap : aligned_positions) {
                if (ap.mandatory) sub_align_indices.insert(ap.align_idx);
            }
            // Then greedy (re-sort and pick)
            if (subs_needed > mandatory_subs) {
                std::vector<const AlignedPos*> optional;
                for (auto& ap : aligned_positions) {
                    if (!ap.mandatory && ap.gain > 1e-12f) {
                        optional.push_back(&ap);
                    }
                }
                std::sort(optional.begin(), optional.end(),
                          [](const AlignedPos* a, const AlignedPos* b) {
                              return a->gain > b->gain;
                          });
                double acc = 0.0;
                double remaining_deficit = static_cast<double>(threshold) - adjusted;
                int extra = 0;
                for (auto* ap : optional) {
                    if (extra >= (subs_needed - mandatory_subs)) break;
                    acc += static_cast<double>(ap->gain);
                    sub_align_indices.insert(ap->align_idx);
                    extra++;
                    if (acc >= remaining_deficit) break;
                }
            }

            // Build window_seq and mutated_seq as an alignment view
            std::string window_seq;
            std::string mutated_seq;
            std::vector<EditInfo> edit_list;
            double score_after = score;

            for (size_t a = 0; a < alignment.size(); ++a) {
                const auto& aop = alignment[a];

                if (aop.op == 'M') {
                    bool do_sub = (sub_align_indices.count(static_cast<int>(a)) > 0);
                    float best_s = -std::numeric_limits<float>::infinity();
                    int best_b = 0;
                    for (int b = 0; b < 4; b++) {
                        float s = pssm[aop.motif_pos].get_log_prob_from_code(b);
                        if (s > best_s) { best_s = s; best_b = b; }
                    }

                    window_seq += aop.base_char;
                    if (do_sub) {
                        EditInfo ei;
                        ei.motif_col = aop.motif_pos + 1; // 1-based
                        ei.ref_base = aop.base_char;
                        ei.alt_base = index_to_base(best_b);
                        ei.gain = col_max_scores[aop.motif_pos] - aop.base_score;
                        ei.edit_type = "sub";
                        edit_list.push_back(ei);
                        mutated_seq += index_to_base(best_b);
                        score_after += static_cast<double>(ei.gain);
                    } else {
                        mutated_seq += aop.base_char;
                    }
                } else if (aop.op == 'I') {
                    // Insertion: motif position has no aligned seq base
                    float best_s = -std::numeric_limits<float>::infinity();
                    int best_b = 0;
                    for (int b = 0; b < 4; b++) {
                        float s = pssm[aop.motif_pos].get_log_prob_from_code(b);
                        if (s > best_s) { best_s = s; best_b = b; }
                    }

                    EditInfo ei;
                    ei.motif_col = aop.motif_pos + 1; // 1-based
                    ei.ref_base = '\0';
                    ei.alt_base = index_to_base(best_b);
                    ei.gain = col_max_scores[aop.motif_pos];
                    ei.edit_type = "ins";
                    edit_list.push_back(ei);
                    window_seq += '-';  // gap in original sequence
                    mutated_seq += index_to_base(best_b);
                    score_after += static_cast<double>(col_max_scores[aop.motif_pos]);
                } else if (aop.op == 'D') {
                    // Deletion: seq base skipped (not aligned to any motif pos)
                    EditInfo ei;
                    ei.motif_col = 0;  // will be None in Python
                    ei.ref_base = aop.base_char;
                    ei.alt_base = '\0';
                    ei.gain = 0.0f;
                    ei.edit_type = "del";
                    edit_list.push_back(ei);
                    window_seq += aop.base_char;  // base present in original
                    mutated_seq += '-';            // gap: base removed
                }
            }

            // Compute score_before: the PWM score of the L-length window
            double sb = 0.0;
            for (int i = 0; i < L; i++) {
                int si = reverse ? (L - 1 - i) : i;
                char b = (si < seq_avail) ? seq_ptr[si] : 'N';
                if (reverse) b = complement_base(b);
                int bi = base_to_index(b);
                if (bi == 4) {
                    double mean_lp = 0.0;
                    for (int bb = 0; bb < 4; bb++)
                        mean_lp += static_cast<double>(pssm[i].get_log_prob_from_code(bb));
                    sb += mean_lp / 4.0;
                } else {
                    sb += static_cast<double>(pssm[i].get_log_prob_from_code(bi));
                }
            }
            float score_before_val = static_cast<float>(sb);

            best_result.n_edits = total_edits;
            best_result.score_before = score_before_val;
            best_result.score_after = static_cast<float>(score_after);
            best_result.edits = edit_list;
            best_result.window_seq = window_seq;
            best_result.mutated_seq = mutated_seq;
        }
    }

    return best_result;
}

WindowResult find_best_window_edits(
    const std::string& seq,
    int roi_start_0, int roi_end_0,
    const DnaPSSM& pssm,
    const std::vector<float>& col_max_scores,
    float S_max,
    float threshold,
    int max_edits,
    int max_indels,
    bool scan_forward, bool scan_reverse,
    float score_min, float score_max,
    const std::vector<float>& gain_values,
    const std::vector<std::vector<uint8_t>>& bin_index)
{
    const int L = pssm.length();
    const bool has_score_min = !std::isnan(score_min);
    const bool has_score_max = !std::isnan(score_max);
    const bool has_score_filter = has_score_min || has_score_max;
    const int seqlen = static_cast<int>(seq.length());

    WindowResult best;
    best.n_edits = -1;
    best.strand = 0;
    best.window_start = 0;

    for (int s0 = roi_start_0; s0 <= roi_end_0; ++s0) {
        const char* window_start = seq.data() + s0;

        auto try_window = [&](bool reverse, int direction) {
            int seq_avail = seqlen - s0;
            // Score filter: compute L-length window score for filtering
            // Only apply if we have at least L bases available
            if (has_score_filter && seq_avail >= L) {
                float logp = 0.0f;
                for (int i = 0; i < L; i++) {
                    int si = reverse ? (L - 1 - i) : i;
                    char base = window_start[si];
                    if (reverse) base = complement_base(base);
                    int bidx = base_to_index(base);
                    if (bidx == 4) {
                        float sum = 0.0f;
                        for (int b = 0; b < 4; b++) sum += pssm[i].get_log_prob_from_code(b);
                        logp += sum / 4.0f;
                    } else {
                        logp += pssm[i].get_log_prob_from_code(bidx);
                    }
                }
                if (has_score_min && logp < score_min) return;
                if (has_score_max && logp > score_max) return;
            }

            WindowResult wr;
            if (max_indels > 0) {
                wr = compute_window_edits_detailed_with_indels(
                    window_start, seq_avail, L, pssm, col_max_scores, S_max,
                    threshold, max_edits, max_indels, reverse);
            } else {
                wr = compute_window_edits_detailed(
                    window_start, L, pssm, col_max_scores, S_max,
                    threshold, max_edits, reverse, gain_values, bin_index);
            }

            if (wr.n_edits < 0) return;

            if (best.n_edits < 0 || wr.n_edits < best.n_edits ||
                (wr.n_edits == best.n_edits && wr.score_before > best.score_before)) {
                best = wr;
                best.strand = direction;
                best.window_start = s0 + 1;
            }
        };

        if (scan_forward) try_window(false, +1);
        if (scan_reverse) try_window(true, -1);
    }

    return best;
}

// Parse PSSM from numpy array (same logic as PMVTrack.cpp parse_pssm)
bool parse_pssm_local(PyObject *obj, DnaPSSM &pssm, double prior) {
    if (!obj || obj == Py_None) return false;

    PMPY arr(PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
    if (!arr || !PyArray_Check((PyArrayObject *)*arr) || PyArray_NDIM((PyArrayObject *)*arr) != 2) {
        PyErr_Clear();
        return false;
    }

    npy_intp rows = PyArray_DIM((PyArrayObject *)*arr, 0);
    npy_intp cols = PyArray_DIM((PyArrayObject *)*arr, 1);
    if (rows <= 0 || cols <= 0) return false;

    npy_intp len = rows;
    bool transposed = false;
    if (cols == 4) {
        len = rows;
        transposed = false;
    } else if (rows == 4) {
        len = cols;
        transposed = true;
    } else {
        return false;
    }

    pssm.resize((int)len);
    for (npy_intp i = 0; i < len; ++i) {
        double pa, pc, pg, pt;
        if (!transposed) {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 0);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 1);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 2);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 3);
        } else {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 0, i);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 1, i);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 2, i);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 3, i);
        }
        pssm[i] = DnaProbVec((float)pa, (float)pc, (float)pg, (float)pt);
    }

    if (prior > 0) {
        pssm.add_dirichlet_prior((float)prior);
    }
    return true;
}

} // anonymous namespace


/*
 * pm_gseq_pwm_edits(seqs, pssm, score_thresh, max_edits, bidirect,
 *                    strand_mode, prior, extend, score_min, score_max,
 *                    max_indels)
 *
 * seqs:         list of str (DNA sequences)
 * pssm:         numpy 2D array (Lx4 or 4xL)
 * score_thresh: float
 * max_edits:    int or None (-1 = no cap)
 * bidirect:     bool
 * strand_mode:  int (-1, 0, 1)
 * prior:        float
 * extend:       bool or int
 * score_min:    float or None
 * score_max:    float or None
 * max_indels:   int (default 0)
 *
 * Returns: dict of lists (columns for DataFrame)
 */
PyObject *pm_gseq_pwm_edits(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    PyObject *py_pssm = nullptr;
    double score_thresh = 0.0;
    PyObject *py_max_edits = nullptr;
    int bidirect = 1;
    int strand_mode = 0;
    double prior = 0.01;
    PyObject *py_extend = nullptr;
    PyObject *py_score_min = nullptr;
    PyObject *py_score_max = nullptr;
    int max_indels = 0;

    if (!PyArg_ParseTuple(args, "OOd|OiidOOOi",
                          &py_seqs, &py_pssm, &score_thresh,
                          &py_max_edits, &bidirect, &strand_mode,
                          &prior, &py_extend, &py_score_min, &py_score_max,
                          &max_indels)) {
        return nullptr;
    }

    try {
        // Parse sequences
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }
        Py_ssize_t n_seqs = PyList_Size(py_seqs);

        // Parse PSSM
        DnaPSSM pssm;
        if (!parse_pssm_local(py_pssm, pssm, prior)) {
            PyErr_SetString(PyExc_ValueError, "pssm must be a numeric array with shape Lx4 or 4xL");
            return nullptr;
        }
        int w = pssm.length();

        float threshold = static_cast<float>(score_thresh);
        if (bidirect) strand_mode = 0;
        bool scan_forward = bidirect || strand_mode >= 0;
        bool scan_reverse = bidirect || strand_mode <= 0;

        int max_edits = -1;
        if (py_max_edits && py_max_edits != Py_None) {
            max_edits = (int)PyLong_AsLong(py_max_edits);
            if (PyErr_Occurred()) return nullptr;
        }

        int extend_val = w - 1;  // default: extend=True
        if (py_extend && py_extend != Py_None) {
            if (PyBool_Check(py_extend)) {
                extend_val = (py_extend == Py_True) ? (w - 1) : 0;
            } else if (PyLong_Check(py_extend)) {
                extend_val = (int)PyLong_AsLong(py_extend);
                if (extend_val < 0) extend_val = 0;
            }
        }

        float score_min = std::numeric_limits<float>::quiet_NaN();
        if (py_score_min && py_score_min != Py_None) {
            score_min = (float)PyFloat_AsDouble(py_score_min);
            if (PyErr_Occurred()) return nullptr;
        }
        float score_max_val = std::numeric_limits<float>::quiet_NaN();
        if (py_score_max && py_score_max != Py_None) {
            score_max_val = (float)PyFloat_AsDouble(py_score_max);
            if (PyErr_Occurred()) return nullptr;
        }

        // Precompute tables
        std::vector<float> col_max_scores(w);
        float S_max = 0.0f;
        for (int i = 0; i < w; i++) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (int b = 0; b < 4; b++) {
                float s = pssm[i].get_log_prob_from_code(b);
                if (s > max_score) max_score = s;
            }
            col_max_scores[i] = max_score;
            S_max += max_score;
        }

        std::set<float, std::greater<float>> unique_gains;
        std::vector<std::vector<uint8_t>> bin_idx(w);
        for (int i = 0; i < w; i++) {
            bin_idx[i].resize(5);
            for (int b = 0; b < 4; b++) {
                float g = col_max_scores[i] - pssm[i].get_log_prob_from_code(b);
                unique_gains.insert(g);
            }
            float min_score = std::numeric_limits<float>::infinity();
            for (int b = 0; b < 4; b++) {
                float s = pssm[i].get_log_prob_from_code(b);
                if (s < min_score) min_score = s;
            }
            unique_gains.insert(col_max_scores[i] - min_score);
        }
        std::vector<float> gain_values(unique_gains.begin(), unique_gains.end());
        for (int i = 0; i < w; i++) {
            for (int b = 0; b < 5; b++) {
                float score;
                if (b < 4) {
                    score = pssm[i].get_log_prob_from_code(b);
                } else {
                    score = std::numeric_limits<float>::infinity();
                    for (int bb = 0; bb < 4; bb++) {
                        float s = pssm[i].get_log_prob_from_code(bb);
                        if (s < score) score = s;
                    }
                }
                float g = col_max_scores[i] - score;
                auto it = std::lower_bound(gain_values.begin(), gain_values.end(),
                                           g, std::greater<float>());
                bin_idx[i][b] = static_cast<uint8_t>(std::distance(gain_values.begin(), it));
            }
        }

        if (max_indels < 0) max_indels = 0;

        // Process each sequence
        struct EditRow {
            int seq_idx;
            int strand;
            int window_start;
            float score_before;
            float score_after;
            int n_edits;
            int edit_num;
            int motif_col;
            char ref;
            char alt;
            float gain;
            std::string edit_type;
            std::string window_seq;
            std::string mutated_seq;
        };

        std::vector<EditRow> all_rows;

        for (Py_ssize_t si = 0; si < n_seqs; si++) {
            PyObject *py_str = PyList_GetItem(py_seqs, si);
            if (!py_str || py_str == Py_None || !PyUnicode_Check(py_str)) continue;

            const char *raw = PyUnicode_AsUTF8(py_str);
            if (!raw) continue;

            std::string seq(raw);
            for (char& c : seq) c = toupper(c);

            int seqlen = static_cast<int>(seq.length());
            // Minimum sequence length: with indels, shortest window is max(1, w-D)
            int min_seq_len = (max_indels > 0) ? std::max(1, w - max_indels) : w;
            if (seqlen < min_seq_len) continue;

            // Compute allowed window starts (0-based)
            int min_window = std::max(1, w - max_indels);
            int start_min0 = 0;
            int start_max0 = seqlen - min_window;
            // For the non-indel path, still need at least w bases
            if (max_indels == 0) {
                start_max0 = std::min(start_max0, seqlen - w);
            }

            if (start_min0 > start_max0) continue;

            WindowResult wr = find_best_window_edits(
                seq, start_min0, start_max0,
                pssm, col_max_scores, S_max,
                threshold, max_edits, max_indels,
                scan_forward, scan_reverse,
                score_min, score_max_val,
                gain_values, bin_idx);

            if (wr.n_edits == 0) {
                EditRow row;
                row.seq_idx = (int)si + 1;
                row.strand = wr.strand;
                row.window_start = wr.window_start;
                row.score_before = wr.score_before;
                row.score_after = wr.score_after;
                row.n_edits = 0;
                row.edit_num = 0;
                row.motif_col = 0;  // will be set to None in Python
                row.ref = 0;
                row.alt = 0;
                row.gain = 0.0f;
                row.edit_type = "";  // will be None in Python
                row.window_seq = wr.window_seq;
                row.mutated_seq = wr.mutated_seq;
                all_rows.push_back(row);
            } else if (wr.n_edits > 0 && !wr.edits.empty()) {
                for (size_t e = 0; e < wr.edits.size(); e++) {
                    EditRow row;
                    row.seq_idx = (int)si + 1;
                    row.strand = wr.strand;
                    row.window_start = wr.window_start;
                    row.score_before = wr.score_before;
                    row.score_after = wr.score_after;
                    row.n_edits = wr.n_edits;
                    row.edit_num = static_cast<int>(e) + 1;
                    row.motif_col = wr.edits[e].motif_col;
                    row.ref = wr.edits[e].ref_base;
                    row.alt = wr.edits[e].alt_base;
                    row.gain = wr.edits[e].gain;
                    row.edit_type = wr.edits[e].edit_type;
                    row.window_seq = wr.window_seq;
                    row.mutated_seq = wr.mutated_seq;
                    all_rows.push_back(row);
                }
            }
        }

        // Build result as dict of lists
        Py_ssize_t n_rows = (Py_ssize_t)all_rows.size();

        PMPY d_seq_idx(PyList_New(n_rows), true);
        PMPY d_strand(PyList_New(n_rows), true);
        PMPY d_wstart(PyList_New(n_rows), true);
        PMPY d_sbefore(PyList_New(n_rows), true);
        PMPY d_safter(PyList_New(n_rows), true);
        PMPY d_nedits(PyList_New(n_rows), true);
        PMPY d_editnum(PyList_New(n_rows), true);
        PMPY d_mcol(PyList_New(n_rows), true);
        PMPY d_ref(PyList_New(n_rows), true);
        PMPY d_alt(PyList_New(n_rows), true);
        PMPY d_gain(PyList_New(n_rows), true);
        PMPY d_etype(PyList_New(n_rows), true);
        PMPY d_wseq(PyList_New(n_rows), true);
        PMPY d_mseq(PyList_New(n_rows), true);

        for (Py_ssize_t i = 0; i < n_rows; i++) {
            const EditRow& row = all_rows[i];
            PyList_SET_ITEM(*d_seq_idx, i, PyLong_FromLong(row.seq_idx));
            PyList_SET_ITEM(*d_strand, i, PyLong_FromLong(row.strand));
            PyList_SET_ITEM(*d_wstart, i, PyLong_FromLong(row.window_start));
            PyList_SET_ITEM(*d_sbefore, i, PyFloat_FromDouble(row.score_before));
            PyList_SET_ITEM(*d_safter, i, PyFloat_FromDouble(row.score_after));
            PyList_SET_ITEM(*d_nedits, i, PyLong_FromLong(row.n_edits));
            PyList_SET_ITEM(*d_editnum, i, PyLong_FromLong(row.edit_num));

            if (row.edit_num == 0) {
                // No edit for this row
                Py_INCREF(Py_None);
                PyList_SET_ITEM(*d_mcol, i, Py_None);
                Py_INCREF(Py_None);
                PyList_SET_ITEM(*d_ref, i, Py_None);
                Py_INCREF(Py_None);
                PyList_SET_ITEM(*d_alt, i, Py_None);
            } else {
                // motif_col: 0 means None (deletions have no motif position)
                if (row.motif_col == 0) {
                    Py_INCREF(Py_None);
                    PyList_SET_ITEM(*d_mcol, i, Py_None);
                } else {
                    PyList_SET_ITEM(*d_mcol, i, PyLong_FromLong(row.motif_col));
                }
                // ref: '\0' means None (insertions have no ref base)
                if (row.ref == 0) {
                    Py_INCREF(Py_None);
                    PyList_SET_ITEM(*d_ref, i, Py_None);
                } else {
                    char buf[2] = {row.ref, '\0'};
                    PyList_SET_ITEM(*d_ref, i, PyUnicode_FromString(buf));
                }
                // alt: '\0' means None (deletions have no alt base)
                if (row.alt == 0) {
                    Py_INCREF(Py_None);
                    PyList_SET_ITEM(*d_alt, i, Py_None);
                } else {
                    char buf[2] = {row.alt, '\0'};
                    PyList_SET_ITEM(*d_alt, i, PyUnicode_FromString(buf));
                }
            }
            PyList_SET_ITEM(*d_gain, i, PyFloat_FromDouble(row.gain));

            // edit_type: empty string means None (n_edits==0 rows)
            if (row.edit_type.empty()) {
                Py_INCREF(Py_None);
                PyList_SET_ITEM(*d_etype, i, Py_None);
            } else {
                PyList_SET_ITEM(*d_etype, i, PyUnicode_FromString(row.edit_type.c_str()));
            }

            PyList_SET_ITEM(*d_wseq, i, PyUnicode_FromString(row.window_seq.c_str()));
            PyList_SET_ITEM(*d_mseq, i, PyUnicode_FromString(row.mutated_seq.c_str()));
        }

        // Build dict
        PyObject *result = PyDict_New();
        PyDict_SetItemString(result, "seq_idx", *d_seq_idx);
        PyDict_SetItemString(result, "strand", *d_strand);
        PyDict_SetItemString(result, "window_start", *d_wstart);
        PyDict_SetItemString(result, "score_before", *d_sbefore);
        PyDict_SetItemString(result, "score_after", *d_safter);
        PyDict_SetItemString(result, "n_edits", *d_nedits);
        PyDict_SetItemString(result, "edit_num", *d_editnum);
        PyDict_SetItemString(result, "motif_col", *d_mcol);
        PyDict_SetItemString(result, "ref", *d_ref);
        PyDict_SetItemString(result, "alt", *d_alt);
        PyDict_SetItemString(result, "gain", *d_gain);
        PyDict_SetItemString(result, "edit_type", *d_etype);
        PyDict_SetItemString(result, "window_seq", *d_wseq);
        PyDict_SetItemString(result, "mutated_seq", *d_mseq);

        return result;

    } catch (std::exception& e) {
        PyErr_SetString(s_pm_err, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(s_pm_err, "pm_gseq_pwm_edits: unknown error");
        return nullptr;
    }
}
