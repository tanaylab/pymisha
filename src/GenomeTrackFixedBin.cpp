#include <errno.h>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "TGLException.h"
#include "GenomeTrackFixedBin.h"
#include "TrackIndex.h"

void GenomeTrackFixedBin::read_interval(const GInterval &interval)
{
	if (m_use_quantile)
		m_sp.reset();

	if (m_functions[MIN_POS])
		m_last_min_pos = numeric_limits<double>::quiet_NaN();
	if (m_functions[EXISTS])
		m_last_exists = 0;
	if (m_functions[SIZE])
		m_last_size = 0;
	if (m_functions[SAMPLE])
		m_last_sample = numeric_limits<float>::quiet_NaN();
	if (m_functions[SAMPLE_POS])
		m_last_sample_pos = numeric_limits<double>::quiet_NaN();
	if (m_functions[FIRST])
		m_last_first = numeric_limits<float>::quiet_NaN();
	if (m_functions[FIRST_POS])
		m_last_first_pos = numeric_limits<double>::quiet_NaN();
	if (m_functions[LAST])
		m_last_last = numeric_limits<float>::quiet_NaN();
	if (m_functions[LAST_POS])
		m_last_last_pos = numeric_limits<double>::quiet_NaN();

	// optimization of the most common case when the expression iterator starts at 0 and steps by bin_size
	if (interval.start == m_cur_coord && interval.end == m_cur_coord + m_bin_size) {
		if (read_next_bin(m_last_avg)) {
			m_cached_bin_idx = (int64_t)(interval.start / m_bin_size);
			m_cached_bin_val = m_last_avg;
			m_cache_valid = true;

			m_last_min = m_last_max = m_last_nearest = m_last_sum = m_last_avg;
			m_last_stddev = numeric_limits<float>::quiet_NaN();
			if (m_functions[MAX_POS])
				m_last_max_pos = interval.start;
			if (m_functions[MIN_POS])
				m_last_min_pos = interval.start;
			if (m_functions[EXISTS])
				m_last_exists = 1;
			if (m_functions[SIZE])
				m_last_size = 1;
			if (m_functions[SAMPLE])
				m_last_sample = m_last_avg;
			if (m_functions[SAMPLE_POS])
				m_last_sample_pos = interval.start;
			if (m_functions[FIRST])
				m_last_first = m_last_avg;
			if (m_functions[FIRST_POS])
				m_last_first_pos = interval.start;
			if (m_functions[LAST])
				m_last_last = m_last_avg;
			if (m_functions[LAST_POS])
				m_last_last_pos = interval.start;
			if (m_use_quantile && !std::isnan(m_last_avg))
				m_sp.add(m_last_avg, s_rnd_func);
		} else {
			m_last_min = m_last_max = m_last_nearest = m_last_avg = m_last_stddev = m_last_sum = numeric_limits<float>::quiet_NaN();
			if (m_functions[MAX_POS])
				m_last_max_pos = numeric_limits<double>::quiet_NaN();
			if (m_functions[MIN_POS])
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
			double overlap_start = std::max(static_cast<double>(sbin * m_bin_size), static_cast<double>(interval.start));
			if (m_functions[MAX_POS])
				m_last_max_pos = overlap_start;
			if (m_functions[MIN_POS])
				m_last_min_pos = overlap_start;
			if (m_functions[EXISTS])
				m_last_exists = 1;
			if (m_functions[SIZE])
				m_last_size = 1;
			if (m_functions[SAMPLE])
				m_last_sample = m_last_avg;
			if (m_functions[SAMPLE_POS])
				m_last_sample_pos = overlap_start;
			if (m_functions[FIRST])
				m_last_first = m_last_avg;
			if (m_functions[FIRST_POS])
				m_last_first_pos = overlap_start;
			if (m_functions[LAST])
				m_last_last = m_last_avg;
			if (m_functions[LAST_POS])
				m_last_last_pos = overlap_start;
			if (m_use_quantile && !std::isnan(m_last_avg))
				m_sp.add(m_last_avg, s_rnd_func);
		} else {
			m_last_min = m_last_max = m_last_nearest = m_last_avg = m_last_stddev = m_last_sum = numeric_limits<float>::quiet_NaN();
			if (m_functions[MAX_POS])
				m_last_max_pos = numeric_limits<double>::quiet_NaN();
			if (m_functions[MIN_POS])
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
		if (m_functions[MAX_POS])
			m_last_max_pos = numeric_limits<double>::quiet_NaN();
		if (m_functions[MIN_POS])
			m_last_min_pos = numeric_limits<double>::quiet_NaN();

		// Bulk read all bins at once into reusable scratch buffer
		m_scratch_bin_vals.clear();
		int64_t bins_read = read_bins_bulk(sbin, ebin - sbin, m_scratch_bin_vals);

		// Precompute which optional reducer groups are active to avoid per-bin work
		const bool need_pos = m_functions[MIN_POS] || m_functions[MAX_POS] ||
		                      m_functions[FIRST_POS] || m_functions[LAST_POS] || m_functions[SAMPLE_POS];
		const bool need_vtrack = m_functions[EXISTS] || m_functions[FIRST] || m_functions[FIRST_POS] ||
		                         m_functions[LAST] || m_functions[LAST_POS] ||
		                         m_functions[SAMPLE] || m_functions[SAMPLE_POS];

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
					if (m_functions[MIN_POS]) {
						double bin_start = static_cast<double>(bin * m_bin_size);
						m_last_min_pos = std::max(bin_start, static_cast<double>(interval.start));
					}
				} else if (m_functions[MIN_POS] && v == m_last_min) {
					double bin_start = static_cast<double>(bin * m_bin_size);
					double candidate_pos = std::max(bin_start, static_cast<double>(interval.start));
					if (std::isnan(m_last_min_pos) || candidate_pos < m_last_min_pos)
						m_last_min_pos = candidate_pos;
				}
				if (v > m_last_max) {
					m_last_max = v;
					if (m_functions[MAX_POS]) {
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

					if (m_functions[EXISTS])
						m_last_exists = 1;

					if (m_functions[FIRST] && std::isnan(m_last_first))
						m_last_first = v;

					if (m_functions[FIRST_POS] && std::isnan(m_last_first_pos))
						m_last_first_pos = overlap_start;

					if (m_functions[LAST])
						m_last_last = v;

					if (m_functions[LAST_POS])
						m_last_last_pos = overlap_start;

					if (m_functions[SAMPLE])
						m_scratch_all_values.push_back(v);
					if (m_functions[SAMPLE_POS])
						m_scratch_all_positions.push_back(overlap_start);
				}

				++num_vs;
				if (m_functions[STDDEV]) {
					const double delta = v - stddev_mean;
					stddev_mean += delta / static_cast<double>(num_vs);
					const double delta2 = v - stddev_mean;
					stddev_m2 += delta * delta2;
				}
			}
		}

		// Finalize size
		if (m_functions[SIZE])
			m_last_size = num_vs;

			// Sample from collected values
			if (m_functions[SAMPLE] && !m_scratch_all_values.empty()) {
				int idx = (int)(s_rnd_func() * m_scratch_all_values.size());
				if (idx >= (int)m_scratch_all_values.size())
					idx = (int)m_scratch_all_values.size() - 1;
				if (idx < 0)
					idx = 0;
				m_last_sample = m_scratch_all_values[idx];
			}

			if (m_functions[SAMPLE_POS] && !m_scratch_all_positions.empty()) {
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
			if (m_functions[MIN_POS])
				m_last_min_pos = numeric_limits<double>::quiet_NaN();
		}

		// Unbiased sample standard deviation via Welford's stable algorithm.
		if (m_functions[STDDEV])
			m_last_stddev = num_vs > 1 ? sqrt(stddev_m2 / static_cast<double>(num_vs - 1))
			                           : numeric_limits<float>::quiet_NaN();
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
	if (num_bins <= 0) {
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

	// Check for indexed format FIRST
	const std::string track_dir = GenomeTrack::get_track_dir(filename);
	const std::string idx_path = track_dir + "/track.idx";

	struct stat idx_st;
	if (stat(idx_path.c_str(), &idx_st) == 0) {
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

		auto idx   = get_track_index(track_dir);
		if (!idx)
			TGLError<GenomeTrackFixedBin>("Failed to load track index for %s", track_dir.c_str());

		auto entry = idx->get_entry(chromid);
		if (!entry || entry->length == 0) {
			// Chromosome not in index or empty contig - treat as empty
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
}

void GenomeTrackFixedBin::init_write(const char *filename, unsigned bin_size, int chromid)
{
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
