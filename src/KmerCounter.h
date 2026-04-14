#ifndef KMER_COUNTER_H_
#define KMER_COUNTER_H_

#include <string>
#include <vector>

#include "GenomeSeqScorer.h"

class KmerCounter : public GenomeSeqScorer
{
public:
    enum CountMode
    {
        SUM,     // Count total occurrences of kmer
        FRACTION // Calculate fraction of kmer in sequence
    };

    KmerCounter(const std::string &kmer, const std::string &genome_root,
                CountMode mode = SUM, bool extend = true, char strand = 0);

    // Constructor with shared GenomeSeqFetch for caching
    KmerCounter(const std::string &kmer, GenomeSeqFetch* shared_seqfetch,
                CountMode mode = SUM, bool extend = true, char strand = 0);

    // Constructor for string-only scoring (no genome database required)
    KmerCounter(const std::string &kmer, CountMode mode = SUM, char strand = 0);

    // Implement the virtual function from the base class
    float score_interval(const GInterval &interval, const GenomeChromKey &chromkey) override;

    // Score a raw string (no genome database needed)
    double count_string(const char* seq, int seq_len);

    // Batch processing accessors
    const std::string &get_kmer() const { return m_kmer; }
    char get_strand() const { return m_strand; }
    bool get_extend() const { return m_extend; }
    CountMode get_mode() const { return m_mode; }

    // Score from pre-found positions (for batch processing)
    float score_from_positions(const std::vector<size_t> &fwd_positions,
                               const std::vector<size_t> &rev_positions,
                               const GInterval &original_interval,
                               const GInterval &fetch_interval) const;

private:
    std::string m_kmer;
    std::string m_kmer_rc;  // reverse complement of m_kmer
    CountMode m_mode;

    // Helper method to count kmers in an interval with a specific strand
    size_t count_in_interval(const GInterval &fetch_interval, const GenomeChromKey &chromkey,
                            const GInterval &original_interval, size_t &count);
};

#endif // KMER_COUNTER_H_