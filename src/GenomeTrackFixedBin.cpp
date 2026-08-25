#include <errno.h>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>

#include "TGLException.h"
#include "GenomeTrackFixedBin.h"
#include "TrackIndex.h"

void GenomeTrackFixedBin::classify_fast_path_mode()
{
	// Escape hatch: force the generic recompute path (benchmarking + safety toggle).
	if (std::getenv("PYMISHA_DISABLE_SLIDING_REDUCER")) {
		m_fast_path_mode = -1;
		return;
	}
	// pymisha computes avg/sum/min/max unconditionally and only registers
	// "extras", so - unlike R, which registers every function - we can only
	// slide when the caller explicitly registered a non-empty subset of the
	// sliding-compatible reducers and nothing else (and no quantile). Readers
	// that register nothing (the common compute-everything case) keep the
	// existing recompute path untouched.
	if (!m_use_quantile && m_func_mask != 0 &&
	    (m_func_mask & ~SLIDING_COMPATIBLE_MASK) == 0)
		m_fast_path_mode = 1;
	else
		m_fast_path_mode = -1;
}

// Incremental sliding-window reducer (port of R misha
// GenomeTrackFixedBin::read_interval_reducers_only). Maintains a Kahan-
// compensated running sum, a non-NaN count, and a RunningLogSumExp across
// consecutive monotone bin windows: when the query advances by `step` bins
// (<= the window width) it pops `step` bins off the front and pushes `step`
// off the back instead of recomputing the whole window. avg/nearest are
// derived from the running sum/count (R derives these in its generic path; we
// unify them here so a single avg/sum/lse/exists/size vtrack all slide).
void GenomeTrackFixedBin::read_interval_reducers_only(const GInterval &interval)
{
	const bool need_lse = has_function(LSE);
	if (!need_lse)
		m_running_lse_initialized = false;

	// Publish from the running window state (multi-bin window).
	auto assign_from_state = [&]() {
		if (m_sliding_num_vs > 0) {
			m_last_sum = (float)m_sliding_sum;
			m_last_avg = m_last_nearest = (float)(m_sliding_sum / (double)m_sliding_num_vs);
			m_last_exists = 1;
			if (need_lse)
				m_last_lse = m_running_lse.window.empty()
					? numeric_limits<float>::quiet_NaN()
					: (float)m_running_lse.value();
		} else {
			m_last_sum = m_last_avg = m_last_nearest = numeric_limits<float>::quiet_NaN();
			m_last_exists = 0;
			if (need_lse)
				m_last_lse = numeric_limits<float>::quiet_NaN();
		}
		m_last_size = (float)m_sliding_num_vs;
		m_last_min = m_last_max = m_last_stddev = numeric_limits<float>::quiet_NaN();
	};

	// Publish for a single-bin read. exists/size are non-NaN-aware (matches R:
	// a single NaN bin has exists=0, size=0), unlike the legacy single-bin
	// recompute path which reported 1 for an in-range NaN bin.
	auto assign_single_value = [&](bool have_value, float v) {
		bool has_num = have_value && !std::isnan(v);
		m_last_sum = has_num ? v : numeric_limits<float>::quiet_NaN();
		m_last_avg = m_last_nearest = has_num ? v : numeric_limits<float>::quiet_NaN();
		m_last_exists = has_num ? 1 : 0;
		m_last_size = has_num ? 1 : 0;
		if (need_lse)
			m_last_lse = has_num ? v : numeric_limits<float>::quiet_NaN();
		m_last_min = m_last_max = m_last_stddev = numeric_limits<float>::quiet_NaN();
	};

	// Common case: iterator advances exactly one dense bin.
	if (interval.start == m_cur_coord && interval.end == m_cur_coord + m_bin_size) {
		float v = numeric_limits<float>::quiet_NaN();
		bool have_value = read_next_bin(v);
		if (have_value) {
			m_cached_bin_idx = (int64_t)(interval.start / m_bin_size);
			m_cached_bin_val = v;
			m_cache_valid = true;
		}
		assign_single_value(have_value, v);
		m_lse_sliding_valid = false;
		m_sliding_sum = 0;
		m_sliding_sum_comp = 0;
		m_sliding_num_vs = 0;
		m_running_lse_initialized = false;
		return;
	}

	int64_t sbin = (int64_t)(interval.start / m_bin_size);
	int64_t ebin = (int64_t)ceil(interval.end / (double)m_bin_size);

	if (ebin == sbin + 1) {
		float v = numeric_limits<float>::quiet_NaN();
		bool have_value = false;

		if (m_cache_valid && m_cached_bin_idx == sbin) {
			v = m_cached_bin_val;
			m_cur_coord = (sbin + 1) * m_bin_size;
			have_value = true;
		} else {
			if (m_cur_coord != sbin * m_bin_size)
				goto_bin(sbin);
			if (read_next_bin(v)) {
				have_value = true;
				m_cached_bin_idx = sbin;
				m_cached_bin_val = v;
				m_cache_valid = true;
			}
		}

		assign_single_value(have_value, v);
		m_lse_sliding_valid = false;
		m_sliding_sum = 0;
		m_sliding_sum_comp = 0;
		m_sliding_num_vs = 0;
		m_running_lse_initialized = false;
		return;
	}

	const int64_t window_size = ebin - sbin;

	// Sliding reducers update when the window shifts forward by <= its width.
	if (m_lse_sliding_valid && window_size > 0 && (!need_lse || m_running_lse_initialized)) {
		int64_t step = sbin - m_lse_prev_sbin;
		int64_t prev_window = m_lse_prev_ebin - m_lse_prev_sbin;
		if (step > 0 && step <= prev_window && window_size == prev_window &&
			(int64_t)m_lse_window_bins.size() == prev_window) {
			int64_t appended = 0;

			if (step == 1) {
				float new_val = numeric_limits<float>::quiet_NaN();
				if (m_cur_coord != m_lse_prev_ebin * m_bin_size)
					goto_bin(m_lse_prev_ebin);
				if (read_next_bin(new_val) && !m_lse_window_bins.empty()) {
					float old_val = m_lse_window_bins.front();
					m_lse_window_bins.pop_front();
					if (!std::isnan(old_val)) {
						kahan_sub_from_sliding_sum(old_val);
						--m_sliding_num_vs;
						if (need_lse)
							m_running_lse.pop_front();
					}

					m_lse_window_bins.push_back(new_val);
					if (!std::isnan(new_val)) {
						kahan_add_to_sliding_sum(new_val);
						++m_sliding_num_vs;
						if (need_lse)
							m_running_lse.push(new_val);
					}
					appended = 1;
				}
			} else {
				m_scratch_bin_vals.clear();
				appended = read_bins_bulk(m_lse_prev_ebin, step, m_scratch_bin_vals);
				auto &new_vals = m_scratch_bin_vals;
				if (appended == step) {
					for (int64_t i = 0; i < step && !m_lse_window_bins.empty(); ++i) {
						float old_val = m_lse_window_bins.front();
						m_lse_window_bins.pop_front();
						if (!std::isnan(old_val)) {
							kahan_sub_from_sliding_sum(old_val);
							--m_sliding_num_vs;
							if (need_lse)
								m_running_lse.pop_front();
						}
					}

					for (int64_t i = 0; i < step; ++i) {
						float new_val = new_vals[i];
						m_lse_window_bins.push_back(new_val);
						if (!std::isnan(new_val)) {
							kahan_add_to_sliding_sum(new_val);
							++m_sliding_num_vs;
							if (need_lse)
								m_running_lse.push(new_val);
						}
					}
				}
			}

			if (appended == step) {
				assign_from_state();
				m_lse_prev_sbin = sbin;
				m_lse_prev_ebin = ebin;
				m_lse_sliding_valid = true;
				m_cached_bin_idx = ebin - 1;
				m_cached_bin_val = m_lse_window_bins.back();
				m_cache_valid = true;
				return;
			}
		}
	}

	// Fallback: full window read + reseed of the sliding state.
	m_scratch_bin_vals.clear();
	int64_t bins_read = read_bins_bulk(sbin, window_size, m_scratch_bin_vals);
	auto &bin_vals = m_scratch_bin_vals;

	if (need_lse)
		m_running_lse.clear();
	m_sliding_sum = 0;
	m_sliding_sum_comp = 0;
	m_sliding_num_vs = 0;
	m_lse_window_bins.clear();
	for (int64_t i = 0; i < bins_read; ++i) {
		float v = bin_vals[i];
		m_lse_window_bins.push_back(v);
		if (!std::isnan(v)) {
			kahan_add_to_sliding_sum(v);
			++m_sliding_num_vs;
			if (need_lse)
				m_running_lse.push(v);
		}
	}
	m_running_lse_initialized = need_lse;

	if (bins_read > 0) {
		m_cached_bin_idx = sbin + bins_read - 1;
		m_cached_bin_val = bin_vals[bins_read - 1];
		m_cache_valid = true;
	}

	assign_from_state();
	m_lse_prev_sbin = sbin;
	m_lse_prev_ebin = ebin;
	m_lse_sliding_valid = bins_read == window_size && window_size > 0;
}

void GenomeTrackFixedBin::read_interval(const GInterval &interval)
{
	if (m_fast_path_mode == 0)
		classify_fast_path_mode();
	if (m_fast_path_mode == 1) {
		read_interval_reducers_only(interval);
		return;
	}

	if (m_use_quantile)
		m_sp.reset();

	if (has_function(MIN_POS))
		m_last_min_pos = numeric_limits<double>::quiet_NaN();
	if (has_function(EXISTS))
		m_last_exists = 0;
	if (has_function(SIZE))
		m_last_size = 0;
	if (has_function(SAMPLE))
		m_last_sample = numeric_limits<float>::quiet_NaN();
	if (has_function(SAMPLE_POS))
		m_last_sample_pos = numeric_limits<double>::quiet_NaN();
	if (has_function(FIRST))
		m_last_first = numeric_limits<float>::quiet_NaN();
	if (has_function(FIRST_POS))
		m_last_first_pos = numeric_limits<double>::quiet_NaN();
	if (has_function(LAST))
		m_last_last = numeric_limits<float>::quiet_NaN();
	if (has_function(LAST_POS))
		m_last_last_pos = numeric_limits<double>::quiet_NaN();

	// optimization of the most common case when the expression iterator starts at 0 and steps by bin_size
	if (interval.start == m_cur_coord && interval.end == m_cur_coord + m_bin_size) {
		if (read_next_bin(m_last_avg)) {
			m_cached_bin_idx = (int64_t)(interval.start / m_bin_size);
			m_cached_bin_val = m_last_avg;
			m_cache_valid = true;

			m_last_min = m_last_max = m_last_nearest = m_last_sum = m_last_avg;
			m_last_stddev = numeric_limits<float>::quiet_NaN();
			if (has_function(LSE))
				m_last_lse = m_last_avg;  // lse of a single value == the value
			if (has_function(MAX_POS))
				m_last_max_pos = interval.start;
			if (has_function(MIN_POS))
				m_last_min_pos = interval.start;
			if (has_function(EXISTS))
				m_last_exists = 1;
			if (has_function(SIZE))
				m_last_size = 1;
			if (has_function(SAMPLE))
				m_last_sample = m_last_avg;
			if (has_function(SAMPLE_POS))
				m_last_sample_pos = interval.start;
			if (has_function(FIRST))
				m_last_first = m_last_avg;
			if (has_function(FIRST_POS))
				m_last_first_pos = interval.start;
			if (has_function(LAST))
				m_last_last = m_last_avg;
			if (has_function(LAST_POS))
				m_last_last_pos = interval.start;
			if (m_use_quantile && !std::isnan(m_last_avg))
				m_sp.add(m_last_avg, s_rnd_func);
		} else {
			m_last_min = m_last_max = m_last_nearest = m_last_avg = m_last_stddev = m_last_sum = numeric_limits<float>::quiet_NaN();
			if (has_function(LSE))
				m_last_lse = numeric_limits<float>::quiet_NaN();
			if (has_function(MAX_POS))
				m_last_max_pos = numeric_limits<double>::quiet_NaN();
			if (has_function(MIN_POS))
				m_last_min_pos = numeric_limits<double>::quiet_NaN();
		}
		return;
	}

	int64_t sbin = (int64_t)(interval.start / m_bin_size);
	int64_t ebin = (int64_t)ceil(interval.end / (double)m_bin_size);

	const bool single_bin = ebin == sbin + 1;
	float cached_val = numeric_limits<float>::quiet_NaN();
	bool use_cache = false;
	bool have_value = false;

	if (single_bin && m_cache_valid && m_cached_bin_idx == sbin) {
		cached_val = m_cached_bin_val;
		use_cache = true;
		have_value = true;
	}

	if (single_bin) {
		if (!use_cache) {
			if (m_cur_coord != sbin * m_bin_size)
				goto_bin(sbin);
			if (read_next_bin(m_last_avg)) {
				m_cached_bin_idx = sbin;
				m_cached_bin_val = m_last_avg;
				m_cache_valid = true;
				have_value = true;
			}
		} else {
			m_last_avg = cached_val;
			// Keep virtual cursor at the end of this bin to match read_next_bin behaviour
			m_cur_coord = (sbin + 1) * m_bin_size;
		}

		if (have_value) {
			m_last_min = m_last_max = m_last_nearest = m_last_sum = m_last_avg;
			m_last_stddev = numeric_limits<float>::quiet_NaN();
			if (has_function(LSE))
				m_last_lse = m_last_avg;  // lse of a single value == the value
			double overlap_start = std::max(static_cast<double>(sbin * m_bin_size), static_cast<double>(interval.start));
			if (has_function(MAX_POS))
				m_last_max_pos = overlap_start;
			if (has_function(MIN_POS))
				m_last_min_pos = overlap_start;
			if (has_function(EXISTS))
				m_last_exists = 1;
			if (has_function(SIZE))
				m_last_size = 1;
			if (has_function(SAMPLE))
				m_last_sample = m_last_avg;
			if (has_function(SAMPLE_POS))
				m_last_sample_pos = overlap_start;
			if (has_function(FIRST))
				m_last_first = m_last_avg;
			if (has_function(FIRST_POS))
				m_last_first_pos = overlap_start;
			if (has_function(LAST))
				m_last_last = m_last_avg;
			if (has_function(LAST_POS))
				m_last_last_pos = overlap_start;
			if (m_use_quantile && !std::isnan(m_last_avg))
				m_sp.add(m_last_avg, s_rnd_func);
		} else {
			m_last_min = m_last_max = m_last_nearest = m_last_avg = m_last_stddev = m_last_sum = numeric_limits<float>::quiet_NaN();
			if (has_function(LSE))
				m_last_lse = numeric_limits<float>::quiet_NaN();
			if (has_function(MAX_POS))
				m_last_max_pos = numeric_limits<double>::quiet_NaN();
			if (has_function(MIN_POS))
				m_last_min_pos = numeric_limits<double>::quiet_NaN();
		}
	} else {
		uint64_t num_vs = 0;
		double stddev_mean = 0;
		double stddev_m2 = 0;

		// Reuse scratch buffers for sampling (avoids per-call allocation)
		m_scratch_all_values.clear();
		m_scratch_all_positions.clear();

		m_last_sum = 0;
		m_last_min = numeric_limits<float>::max();
		m_last_max = -numeric_limits<float>::max();
		if (has_function(MAX_POS))
			m_last_max_pos = numeric_limits<double>::quiet_NaN();
		if (has_function(MIN_POS))
			m_last_min_pos = numeric_limits<double>::quiet_NaN();

		// Bulk read all bins at once into reusable scratch buffer
		m_scratch_bin_vals.clear();
		int64_t bins_read = read_bins_bulk(sbin, ebin - sbin, m_scratch_bin_vals);

		// Precompute which optional reducer groups are active to avoid per-bin work
		const bool need_pos = has_function(MIN_POS) || has_function(MAX_POS) ||
		                      has_function(FIRST_POS) || has_function(LAST_POS) || has_function(SAMPLE_POS);
		const bool need_vtrack = has_function(EXISTS) || has_function(FIRST) || has_function(FIRST_POS) ||
		                         has_function(LAST) || has_function(LAST_POS) ||
		                         has_function(SAMPLE) || has_function(SAMPLE_POS);

		for (int64_t i = 0; i < bins_read; ++i) {
			int64_t bin = sbin + i;
			float v = m_scratch_bin_vals[i];

			m_cached_bin_idx = bin;
			m_cached_bin_val = v;
			m_cache_valid = true;

			if (!std::isnan(v)) {
				m_last_sum += v;

				// Min/max are always tracked (callers read them unconditionally)
				// Only position bookkeeping is conditional
				if (v < m_last_min) {
					m_last_min = v;
					if (has_function(MIN_POS)) {
						double bin_start = static_cast<double>(bin * m_bin_size);
						m_last_min_pos = std::max(bin_start, static_cast<double>(interval.start));
					}
				} else if (has_function(MIN_POS) && v == m_last_min) {
					double bin_start = static_cast<double>(bin * m_bin_size);
					double candidate_pos = std::max(bin_start, static_cast<double>(interval.start));
					if (std::isnan(m_last_min_pos) || candidate_pos < m_last_min_pos)
						m_last_min_pos = candidate_pos;
				}
				if (v > m_last_max) {
					m_last_max = v;
					if (has_function(MAX_POS)) {
						double bin_start = static_cast<double>(bin * m_bin_size);
						m_last_max_pos = std::max(bin_start, static_cast<double>(interval.start));
					}
				}

				if (m_use_quantile)
					m_sp.add(v, s_rnd_func);

				if (need_vtrack) {
					double overlap_start = 0;
					if (need_pos) {
						double bin_start = static_cast<double>(bin * m_bin_size);
						overlap_start = std::max(bin_start, static_cast<double>(interval.start));
					}

					if (has_function(EXISTS))
						m_last_exists = 1;

					if (has_function(FIRST) && std::isnan(m_last_first))
						m_last_first = v;

					if (has_function(FIRST_POS) && std::isnan(m_last_first_pos))
						m_last_first_pos = overlap_start;

					if (has_function(LAST))
						m_last_last = v;

					if (has_function(LAST_POS))
						m_last_last_pos = overlap_start;

					if (has_function(SAMPLE))
						m_scratch_all_values.push_back(v);
					if (has_function(SAMPLE_POS))
						m_scratch_all_positions.push_back(overlap_start);
				}

				++num_vs;
				if (has_function(STDDEV)) {
					const double delta = v - stddev_mean;
					stddev_mean += delta / static_cast<double>(num_vs);
					const double delta2 = v - stddev_mean;
					stddev_m2 += delta * delta2;
				}
			}
		}

		// Finalize size
		if (has_function(SIZE))
			m_last_size = num_vs;

		// Sample from collected values
		if (has_function(SAMPLE) && !m_scratch_all_values.empty()) {
			int idx = (int)(s_rnd_func() * m_scratch_all_values.size());
			if (idx >= (int)m_scratch_all_values.size())
				idx = (int)m_scratch_all_values.size() - 1;
			if (idx < 0)
				idx = 0;
			m_last_sample = m_scratch_all_values[idx];
		}

		if (has_function(SAMPLE_POS) && !m_scratch_all_positions.empty()) {
			int idx = (int)(s_rnd_func() * m_scratch_all_positions.size());
			if (idx >= (int)m_scratch_all_positions.size())
				idx = (int)m_scratch_all_positions.size() - 1;
			if (idx < 0)
				idx = 0;
			m_last_sample_pos = m_scratch_all_positions[idx];
		}

		if (num_vs > 0)
			m_last_avg = m_last_nearest = m_last_sum / num_vs;
		else {
			m_last_avg = m_last_nearest = m_last_min = m_last_max = m_last_sum = numeric_limits<float>::quiet_NaN();
			if (has_function(MIN_POS))
				m_last_min_pos = numeric_limits<double>::quiet_NaN();
		}

		// Unbiased sample standard deviation via Welford's stable algorithm.
		if (has_function(STDDEV))
			m_last_stddev = num_vs > 1 ? sqrt(stddev_m2 / static_cast<double>(num_vs - 1))
			                           : numeric_limits<float>::quiet_NaN();

		// LSE on the generic fallback path. Normal operation slides LSE in
		// read_interval_reducers_only; this keeps PYMISHA_DISABLE_SLIDING_REDUCER
		// (and any future mode -1 LSE reader) correct, matching R's generic path.
		// inf was already coerced to NaN by read_bins_bulk, so a non-empty window
		// has a finite max.
		if (has_function(LSE)) {
			if (num_vs > 0) {
				double m = -numeric_limits<double>::infinity();
				for (int64_t i = 0; i < bins_read; ++i) {
					float v = m_scratch_bin_vals[i];
					if (!std::isnan(v) && (double)v > m)
						m = (double)v;
				}
				double sum_exp = 0.0;
				for (int64_t i = 0; i < bins_read; ++i) {
					float v = m_scratch_bin_vals[i];
					if (!std::isnan(v))
						sum_exp += std::exp((double)v - m);
				}
				m_last_lse = (float)(m + std::log(sum_exp));
			} else {
				m_last_lse = numeric_limits<float>::quiet_NaN();
			}
		}
	}
}

double GenomeTrackFixedBin::last_max_pos() const
{
	return m_last_max_pos;
}

double GenomeTrackFixedBin::last_min_pos() const
{
	return m_last_min_pos;
}

int64_t GenomeTrackFixedBin::read_bins_bulk(int64_t start_bin, int64_t num_bins, std::vector<float> &vals)
{
	if (num_bins <= 0 || start_bin < 0) {
		// start_bin < 0 would memcpy before the mmap. Callers must verify
		// interval coordinates; this is the backstop.
		vals.clear();
		return 0;
	}

	// Clamp to available samples
	int64_t available = m_num_samples - start_bin;
	if (available <= 0) {
		vals.clear();
		return 0;
	}
	int64_t to_read = std::min(num_bins, available);

	vals.resize(to_read);

	if (m_mmap_data) {
		// mmap path: direct memcpy from mapped region
		memcpy(vals.data(), m_mmap_data + start_bin, to_read * sizeof(float));
		m_cur_bin = start_bin + to_read;
	} else {
		if (m_cur_coord != start_bin * m_bin_size)
			goto_bin(start_bin);

		// Bulk read all bins in one syscall
		size_t bytes_to_read = to_read * sizeof(float);
		uint64_t bytes_read = m_bfile.read(vals.data(), bytes_to_read);

		if (bytes_read != bytes_to_read) {
			if (m_bfile.error())
				TGLError<GenomeTrackFixedBin>("Failed to read a dense track file %s: %s", m_bfile.file_name().c_str(), strerror(errno));
			// Partial read - adjust size
			to_read = bytes_read / sizeof(float);
			vals.resize(to_read);
		}
	}

	// Convert infinity to NaN (matching read_next_bin behavior)
	for (int64_t i = 0; i < to_read; ++i) {
		if (std::isinf(vals[i]))
			vals[i] = numeric_limits<float>::quiet_NaN();
	}

	// Update cursor position
	m_cur_coord = (start_bin + to_read) * m_bin_size;

	return to_read;
}

void GenomeTrackFixedBin::read_header_at_current_pos_(BufferedFile &bf)
{
	int32_t signature = 0;
	if (bf.read(&signature, sizeof(signature)) != sizeof(signature) || signature <= 0)
		TGLError<GenomeTrackFixedBin>("Invalid fixed-bin header in %s", bf.file_name().c_str());
	if (bf.read(&m_bin_size, sizeof(m_bin_size)) != sizeof(m_bin_size))
		TGLError<GenomeTrackFixedBin>("Invalid fixed-bin header in %s", bf.file_name().c_str());
}

void GenomeTrackFixedBin::init_read(const char *filename, const char *mode, int chromid)
{
	m_base_offset = 0; // Reset for per-chromosome
	m_cur_coord = 0;
	uint64_t header_start = 0;
	uint64_t total_bytes = 0;
	m_cached_bin_idx = -1;
	m_cached_bin_val = numeric_limits<float>::quiet_NaN();
	m_cache_valid = false;
	reset_sliding_state();

	// Check for indexed format FIRST.
	// E.1.4: get_track_index() already caches loaded indexes and returns
	// nullptr silently when no track.idx exists (TrackIndex::load returns
	// false on ENOENT). Calling it directly avoids a redundant per-chrom
	// stat(track.idx) syscall on the hot read path; critical for
	// million-contig genomes on NFS.
	const std::string track_dir = GenomeTrack::get_track_dir(filename);
	auto idx = get_track_index(track_dir);

	if (idx) {
		// --- INDEXED PATH ---
		const std::string dat_path  = track_dir + "/track.dat";

		// Reopen file if: not open, path changed, or mode changed
		if (!m_dat_open || m_dat_path != dat_path || m_dat_mode != mode) {
			m_bfile.close();
			if (m_bfile.open(dat_path.c_str(), mode))
				TGLError<GenomeTrackFixedBin>("Cannot open %s: %s", dat_path.c_str(), strerror(errno));
			m_dat_open = true;
			m_dat_path = dat_path;
			m_dat_mode = mode;
		}

		auto entry = idx->get_entry(chromid);
		if (!entry || entry->length == 0) {
			// Chromosome not in index or empty contig - treat as empty.
			// But still populate m_bin_size from the first non-empty entry,
			// otherwise gtrack.info on an indexed track whose probed chrom has
			// no data (e.g. produced by a per-chrom pack with a destination
			// chrom not in the source's per-chrom file list) reports
			// bin_size = 0, and subsequent unguarded `/ m_bin_size` divisions
			// in the read paths trigger SIGFPE.
			if (m_bin_size == 0) {
				for (const auto &e : idx->get_all_entries()) {
					if (e.length == 0) continue;
					if (m_bfile.seek(e.offset, SEEK_SET))
						TGLError<GenomeTrackFixedBin>("Failed to seek to offset %llu in %s",
							(unsigned long long)e.offset, dat_path.c_str());
					if (m_bfile.read(&m_bin_size, sizeof(m_bin_size)) != sizeof(m_bin_size))
						TGLError<GenomeTrackFixedBin>("Invalid fixed-bin header in %s", dat_path.c_str());
					break;
				}
			}
			m_num_samples = 0;
			m_chromid = chromid;
			return;
		}

		if (m_bfile.seek(entry->offset, SEEK_SET))
			TGLError<GenomeTrackFixedBin>("Failed to seek to offset %llu in %s",
				(unsigned long long)entry->offset, dat_path.c_str());

		header_start = entry->offset;
		// For indexed format, read just bin_size (no signature)
		// The data was copied as-is from per-chromosome files which have: bin_size + values
		if (m_bfile.read(&m_bin_size, sizeof(m_bin_size)) != sizeof(m_bin_size))
			TGLError<GenomeTrackFixedBin>("Invalid fixed-bin header in %s", dat_path.c_str());

		m_base_offset = entry->offset; 
		total_bytes = entry->length;
	} else {
		// --- PER-CHROMOSOME PATH ---
		m_bfile.close();
		m_dat_open = false;

		if (m_bfile.open(filename, mode))
			TGLError<GenomeTrackFixedBin>("%s", strerror(errno));

		if (m_bfile.read(&m_bin_size, sizeof(m_bin_size)) != sizeof(m_bin_size)) {
			if (m_bfile.error())
				TGLError<GenomeTrackFixedBin>("Failed to read a dense track file %s: %s", filename, strerror(errno));
			TGLError<GenomeTrackFixedBin>("Invalid format of a dense track file %s", filename);
		}

		header_start = 0;
		total_bytes = m_bfile.file_size();
	}

	// --- COMMON LOGIC ---
	const uint64_t header_size = m_bfile.tell() - header_start;
	if (total_bytes < header_size || m_bin_size <= 0)
		TGLError<GenomeTrackFixedBin>("Invalid format of a dense track file %s", filename);
	const uint64_t data_bytes = total_bytes - header_size;
	if (data_bytes % sizeof(float) != 0)
		TGLError<GenomeTrackFixedBin>("Invalid format of a dense track file %s", filename);

	m_num_samples = (int64_t)(data_bytes / sizeof(float));
	m_chromid = chromid;

	// Set up mmap for read-only mode (naryn pattern)
	// For indexed format: reuse existing mmap if same file (avoid re-mmap per chromosome)
	m_mmap_data = nullptr;
	m_mmap_num_bins = 0;
	m_cur_bin = 0;

	if (strcmp(mode, "rb") == 0 && m_num_samples > 0) {
		const std::string file_path = m_dat_open ? m_dat_path : std::string(filename);

		// Only re-mmap if file changed (indexed format reuses same track.dat)
		if (!m_mmap.is_open() || m_mmap_path != file_path) {
			m_mmap.close();
			m_mmap.open(file_path, true /* sequential */);
			m_mmap_path = file_path;
		}

		if (m_mmap.is_open()) {
			const uint64_t data_offset = m_base_offset + sizeof(m_bin_size);
			if (data_offset + m_num_samples * sizeof(float) <= m_mmap.size()) {
				m_mmap_data = reinterpret_cast<const float *>(m_mmap.data() + data_offset);
				m_mmap_num_bins = m_num_samples;
			} else {
				m_mmap.close();  // file too small, fall back to BufferedFile
				m_mmap_path.clear();
			}
		}
	} else {
		// Write/update mode or empty: close any existing mmap
		if (m_mmap.is_open()) {
			m_mmap.close();
			m_mmap_path.clear();
		}
	}
}

void GenomeTrackFixedBin::init_write(const char *filename, unsigned bin_size, int chromid)
{
	// open() below drops whatever is still open, and fclose() is where a deferred
	// ENOSPC/EDQUOT surfaces - with nothing left to report it to. Flush the
	// previous chromosome's file while a failure can still be raised. The last
	// chromosome of a run is the caller's flush_writes() after the loop.
	flush_writes();

	m_num_samples = 0;
	m_cur_coord = 0;

	const mode_t old_umask = umask(07);

	if (m_bfile.open(filename, "wb")) {
		umask(old_umask);
		TGLError<GenomeTrackFixedBin>("Opening a dense track file %s: %s", filename, strerror(errno));
	}
	umask(old_umask);

	m_bin_size = bin_size;
	if (m_bfile.write(&m_bin_size, sizeof(m_bin_size)) != sizeof(m_bin_size)) {
		if (m_bfile.error())
			TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s: %s", filename, strerror(errno));
		TGLError<GenomeTrackFixedBin>("Failed to write a dense track file %s", filename);
	}

	m_chromid = chromid;
}
