/*
 * BinsManager.h
 *
 * Multi-dimensional bin manager for gdist.
 * Manages multiple BinFinder instances and maps N-dimensional values to 1D indices.
 */

#ifndef BINSMANAGER_H_
#define BINSMANAGER_H_

#include <cmath>
#include <vector>
#include <string>

#include "BinFinder.h"
#include "TGLException.h"

// BinsManager manages multiple BinFinder instances for N-dimensional distributions
// Converts N values to a single flat index into the distribution array
class BinsManager {
public:
    BinsManager() : m_totalbins(0), m_include_lowest(false) {}

    // Initialize with a vector of break vectors and include_lowest flag
    void init(const std::vector<std::vector<double>>& breaks_list, bool include_lowest);

    // Returns -1 if any of the values does not fall into a bin
    int vals2idx(const std::vector<double>& vals) const;

    // Convert flat index back to N-dimensional bin indices
    std::vector<int> idx2bins(int idx) const;

    bool             get_include_lowest() const { return m_include_lowest; }
    uint64_t         get_total_bins() const { return m_totalbins; }
    unsigned         get_num_bin_finders() const { return m_bin_finders.size(); }
    const BinFinder& get_bin_finder(int idx) const { return m_bin_finders[idx]; }

    // Get bin labels for a dimension (returns strings like "(0,0.2]", "(0.2,0.5]", etc.)
    std::vector<std::string> get_bin_labels(unsigned dim) const;

private:
    std::vector<BinFinder> m_bin_finders;
    std::vector<uint64_t>  m_track_mult;   // multipliers for flat index calculation
    uint64_t               m_totalbins;
    bool                   m_include_lowest;
};


//-------------------------------------- IMPLEMENTATION -------------------------------------------

inline void BinsManager::init(const std::vector<std::vector<double>>& breaks_list, bool include_lowest)
{
    unsigned num_break_sets = breaks_list.size();

    m_include_lowest = include_lowest;
    m_bin_finders.clear();
    m_bin_finders.reserve(num_break_sets);
    m_track_mult.resize(num_break_sets);
    m_totalbins = 1;

    for (unsigned i = 0; i < num_break_sets; ++i) {
        const auto& breaks = breaks_list[i];

        if (breaks.size() < 2)
            TGLError<BinsManager>("breaks[%d] must have at least 2 elements", i);

        m_bin_finders.push_back(BinFinder());
        m_bin_finders.back().init(breaks, m_include_lowest);

        m_totalbins *= m_bin_finders.back().get_numbins();
        m_track_mult[i] = (i == 0) ? 1 : m_track_mult[i - 1] * m_bin_finders[i - 1].get_numbins();
    }
}

inline int BinsManager::vals2idx(const std::vector<double>& vals) const
{
    int res = 0;

    for (size_t i = 0; i < vals.size(); ++i) {
        if (std::isnan(vals[i]))
            return -1;

        int bin = m_bin_finders[i].val2bin(vals[i]);

        if (bin < 0)
            return -1;

        res += bin * m_track_mult[i];
    }

    return res;
}

inline std::vector<int> BinsManager::idx2bins(int idx) const
{
    std::vector<int> bins(m_bin_finders.size());

    // Reverse of vals2idx: extract each dimension's bin from flat index
    for (int i = (int)m_bin_finders.size() - 1; i >= 0; --i) {
        unsigned numbins = m_bin_finders[i].get_numbins();
        bins[i] = (idx / m_track_mult[i]) % numbins;
    }

    return bins;
}

inline std::vector<std::string> BinsManager::get_bin_labels(unsigned dim) const
{
    const BinFinder& bf = m_bin_finders[dim];
    const auto& breaks = bf.get_breaks();
    unsigned numbins = bf.get_numbins();

    std::vector<std::string> labels;
    labels.reserve(numbins);

    for (unsigned j = 0; j < numbins; ++j) {
        char buf[256];
        // First bin may be closed if include_lowest
        const char* left_bracket = (j == 0 && m_include_lowest) ? "[" : "(";
        snprintf(buf, sizeof(buf), "%s%g,%g]", left_bracket, breaks[j], breaks[j + 1]);
        labels.push_back(buf);
    }

    return labels;
}

#endif /* BINSMANAGER_H_ */
