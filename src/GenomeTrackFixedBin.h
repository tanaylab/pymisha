/*
 * GenomeTrackFixedBin.h
 *
 *  Created on: May 15, 2011
 *      Author: hoichman
 */

#ifndef GENOMETRACKFIXEDBIN_H_
#define GENOMETRACKFIXEDBIN_H_

#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <deque>

#include "GenomeTrack1D.h"
#include "MmapFile.h"
#include "utils/RunningLogSumExp.h"

// !!!!!!!!! IN CASE OF ERROR THIS CLASS THROWS TGLException  !!!!!!!!!!!!!!!!

class GenomeTrackFixedBin : public GenomeTrack1D {
public:
	GenomeTrackFixedBin() : GenomeTrack1D(FIXED_BIN), m_bin_size(0), m_num_samples(0), m_cur_coord(0), m_last_min_pos(numeric_limits<double>::quiet_NaN()) {}

	void read_interval(const GInterval &interval) override;
	double last_max_pos() const override;
	double last_min_pos() const override;

	void init_read(const char *filename, int chromid) { init_read(filename, "rb", chromid); }
	void init_write(const char *filename, unsigned bin_size, int chromid);

	void init_update(const char *filename, int chromid) { init_read(filename, "rb+", chromid); }

	// Pushes the stdio buffer out and reports the failure a plain write() cannot:
	// a dense track file finalised without this can be silently truncated on a
	// full disk. Ported from misha 5.11.22.
	void flush_writes()
	{
		if (m_bfile.opened() && m_bfile.flush())
			TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
	}

	unsigned get_bin_size() const { return m_bin_size; }
	int64_t  get_num_samples() const { return m_num_samples; }

	void goto_bin(uint64_t bin);

	bool read_next_bin(float &val);
	// Bulk read multiple bins into buffer, returns number of bins actually read
	int64_t read_bins_bulk(int64_t start_bin, int64_t num_bins, std::vector<float> &vals);
	void write_next_bin(float val);
	void write_next_bins(float *vals, uint64_t num_vals);

protected:
	unsigned  m_bin_size;
	int64_t   m_num_samples;
	int64_t   m_cur_coord;
	double    m_last_min_pos;
	int64_t   m_base_offset{0};
	int64_t   m_cached_bin_idx{-1};
	float     m_cached_bin_val{numeric_limits<float>::quiet_NaN()};
	bool      m_cache_valid{false};

	// ---- incremental sliding-window reducer state (R misha mode-1 port) ----
	// A reader slides only when it registered a non-empty subset of these funcs
	// and nothing else (and no quantile). pymisha computes avg/sum/min/max
	// unconditionally, so callers must explicitly register their reducer to opt
	// into sliding; readers that register nothing keep the compute-everything
	// path unchanged. See classify_fast_path_mode().
	static constexpr uint32_t SLIDING_COMPATIBLE_MASK =
		(1u << AVG) | (1u << NEAREST) | (1u << SUM) |
		(1u << LSE) | (1u << EXISTS) | (1u << SIZE);

	int       m_fast_path_mode{0};   // 0=unclassified, 1=sliding reducers, -1=generic
	RunningLogSumExp m_running_lse;
	std::deque<float> m_lse_window_bins;   // raw window values incl. NaN placeholders
	int64_t   m_lse_prev_sbin{-1};
	int64_t   m_lse_prev_ebin{-1};
	double    m_sliding_sum{0.0};
	double    m_sliding_sum_comp{0.0};     // Kahan compensation
	int64_t   m_sliding_num_vs{0};         // count of non-NaN values in window
	bool      m_lse_sliding_valid{false};
	bool      m_running_lse_initialized{false};

	void classify_fast_path_mode();
	void read_interval_reducers_only(const GInterval &interval);

	inline void kahan_add_to_sliding_sum(double value) {
		double y = value - m_sliding_sum_comp;
		double t = m_sliding_sum + y;
		m_sliding_sum_comp = (t - m_sliding_sum) - y;
		m_sliding_sum = t;
	}
	inline void kahan_sub_from_sliding_sum(double value) { kahan_add_to_sliding_sum(-value); }
	inline void reset_sliding_state() {
		m_fast_path_mode = 0;
		m_running_lse.clear();
		m_lse_window_bins.clear();
		m_lse_prev_sbin = m_lse_prev_ebin = -1;
		m_sliding_sum = m_sliding_sum_comp = 0.0;
		m_sliding_num_vs = 0;
		m_lse_sliding_valid = false;
		m_running_lse_initialized = false;
	}

	// Reusable scratch buffers for multi-bin path (avoids per-call allocation)
	std::vector<float> m_scratch_bin_vals;
	std::vector<float> m_scratch_all_values;
	std::vector<double> m_scratch_all_positions;

	// State for indexed "smart handle"
	std::string m_dat_path;
	std::string m_dat_mode;
	bool        m_dat_open{false};

	// mmap-backed read path (naryn pattern): pointer dereference instead of fread
	MmapFile m_mmap;
	std::string m_mmap_path;  // track which file is mmap'd (avoid re-mmap on chrom switch)
	const float *m_mmap_data{nullptr};  // points to first bin value in mmap'd region
	int64_t m_mmap_num_bins{0};
	int64_t m_cur_bin{0};  // current bin index for mmap path

	void init_read(const char *filename, const char *mode, int chromid);

	// Helper to parse header at current file position
	void read_header_at_current_pos_(BufferedFile &bf);
};


//------------------------------ IMPLEMENTATION ------------------------------------

inline void GenomeTrackFixedBin::goto_bin(uint64_t bin)
{
	if (m_mmap_data) {
		m_cur_bin = bin;
	} else {
		// Add m_base_offset to the absolute seek for indexed format support
		if (m_bfile.seek((long)(m_base_offset + sizeof(m_bin_size) + (uint64_t)bin * sizeof(float)), SEEK_SET))
			TGLError<GenomeTrackFixedBin>("Failed to seek a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
	}
	m_cur_coord = bin * m_bin_size;
}


inline bool GenomeTrackFixedBin::read_next_bin(float &val)
{
	if (m_mmap_data) {
		if (m_cur_bin >= m_mmap_num_bins)
			return false;
		val = m_mmap_data[m_cur_bin++];
		if (isinf(val))
			val = numeric_limits<float>::quiet_NaN();
		m_cur_coord += m_bin_size;
		return true;
	}

	if (m_bfile.read(&val, sizeof(val)) != sizeof(val)) {
		if (m_bfile.error())
			TGLError<GenomeTrackFixedBin>("Failed to read a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
		return false;
	}

	if (isinf(val))
		val = numeric_limits<float>::quiet_NaN();

	m_cur_coord += m_bin_size;
	return true;
}

inline void GenomeTrackFixedBin::write_next_bin(float val)
{
	if (m_bfile.write(&val, sizeof(val)) != sizeof(val)) {
		if (m_bfile.error())
			TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
		TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s", m_bfile.file_name().c_str());
	}
	m_num_samples++;
	m_cur_coord += m_bin_size;
}

inline void GenomeTrackFixedBin::write_next_bins(float *vals, uint64_t num_vals)
{
	uint64_t size = sizeof(vals[0]) * num_vals;
	if (m_bfile.write(vals, size) != size) {
		if (m_bfile.error())
			TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
		TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s", m_bfile.file_name().c_str());
	}
	m_num_samples += num_vals;
	m_cur_coord += m_bin_size * num_vals;
}

#endif /* GENOMETRACKFIXEDBIN_H_ */
