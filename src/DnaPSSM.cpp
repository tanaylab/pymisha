#include <cstdint>
#include "port.h"
#include "DnaPSSM.h"
#include <limits>


const float NEG_INF = -std::numeric_limits<float>::infinity();

// O(1) lookup tables for base encoding — replaces switch statements in hot loops.
// BASE_ENCODE:      A/a->0, C/c->1, G/g->2, T/t->3, else->-1
// COMPLEMENT_ENCODE: A/a->3(T), C/c->2(G), G/g->1(C), T/t->0(A), else->-1
// NEUTRAL_CHAR:     1 for N/n/*, 0 otherwise

#define X -1
const int8_t DnaLookupTables::BASE_ENCODE[256] = {
//  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 0-15
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 16-31
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 32-47
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 48-63
    X,  0,  X,  1,  X,  X,  X,  2,  X,  X,  X,  X,  X,  X,  X,  X,  // 64-79  (A=65->0, C=67->1, G=71->2)
    X,  X,  X,  X,  3,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 80-95  (T=84->3)
    X,  0,  X,  1,  X,  X,  X,  2,  X,  X,  X,  X,  X,  X,  X,  X,  // 96-111 (a=97->0, c=99->1, g=103->2)
    X,  X,  X,  X,  3,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 112-127 (t=116->3)
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 128-143
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 144-159
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 160-175
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 176-191
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 192-207
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 208-223
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 224-239
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // 240-255
};

const int8_t DnaLookupTables::COMPLEMENT_ENCODE[256] = {
//  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  3,  X,  2,  X,  X,  X,  1,  X,  X,  X,  X,  X,  X,  X,  X,  // A->3(T), C->2(G), G->1(C)
    X,  X,  X,  X,  0,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // T->0(A)
    X,  3,  X,  2,  X,  X,  X,  1,  X,  X,  X,  X,  X,  X,  X,  X,  // a->3, c->2, g->1
    X,  X,  X,  X,  0,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  // t->0
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
    X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,  X,
};
#undef X

const int8_t DnaLookupTables::NEUTRAL_CHAR[256] = {
//  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  // '*'=42->1
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  // 'N'=78->1
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  // 'n'=110->1
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
};


void DnaProbVec::normalize()
{
	float sum = m_p[0] + m_p[1] + m_p[2] + m_p[3];

	m_p[0] /= sum;
	m_p[1] /= sum;
	m_p[2] /= sum;
	m_p[3] /= sum;

	if(m_p[0] != 0) {
		m_logp[0] = log(m_p[0]);
	} else {
		m_logp[0] = -std::numeric_limits<float>::infinity();
	}
	if(m_p[1] != 0) {
		m_logp[1] = log(m_p[1]);
	} else {
		m_logp[1] = -std::numeric_limits<float>::infinity();
	}
	if(m_p[2] != 0) {
		m_logp[2] = log(m_p[2]);
	} else {
		m_logp[2] = -std::numeric_limits<float>::infinity();
	}
	if(m_p[3] != 0) {
		m_logp[3] = log(m_p[3]);
	} else {
		m_logp[3] = -std::numeric_limits<float>::infinity();
	}
}


float DnaProbVec::get_max_log_prob() const
{
	float logp = m_logp[0];
	if(m_logp[1] > logp) {
		logp = m_logp[1];
	}
	if(m_logp[2] > logp) {
		logp = m_logp[2];
	}
	if(m_logp[3] > logp) {
		logp = m_logp[3];
	}

	return(logp);
}


const DnaPSSM &DnaPSSM::operator=(const DnaPSSM &other)
{
	m_chars = other.m_chars;
	m_min_range = other.m_min_range;
	m_max_range = other.m_max_range;
	m_bidirect = other.m_bidirect;
	return(*this);
}
void DnaPSSM::resize(int sz)
{
	m_chars.resize(sz);
}



void DnaPSSM::calc_like(const string &target, float &logp) const
{

	string::const_iterator i = target.begin();
	logp = 0;
	for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
	    p < m_chars.end();
	    p++) {
		int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*i];
		if(code < 0) {
			logp = NEG_INF;
			return;
		}
		logp += p->get_log_prob_from_code(code);
		i++;
	}
}
void DnaPSSM::calc_like_rc(const string &target, float &logp) const
{

	string::const_iterator i = target.begin();
	logp = 0;
	for(vector<DnaProbVec>::const_reverse_iterator p = m_chars.rbegin();
	    p != m_chars.rend();
	    p++) {
		int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*i];
		if(code < 0) {
			logp = NEG_INF;
			return;
		}
		logp += p->get_log_prob_from_code(code);
		i++;
	}
}
void DnaPSSM::calc_like(string::const_iterator &j, float &logp) const
{

	logp = 0;
	string::const_iterator i = j;
	for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
	    p < m_chars.end();
	    p++) {
		int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*i];
		if(code < 0) {
			logp = NEG_INF;
			return;
		}
		logp += p->get_log_prob_from_code(code);
		i++;
	}
}

static const float c_log_quarter = -1.38629;

void DnaPSSM::calc_like_rc(string::const_iterator &j, float &logp) const
{

	logp = 0;
	string::const_iterator i = j;
	for(vector<DnaProbVec>::const_reverse_iterator p = m_chars.rbegin();
	    p != m_chars.rend();
	    p++) {
		int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*i];
		if(code < 0) {
			logp = NEG_INF;
			return;
		}
		logp += p->get_log_prob_from_code(code);
		i++;
	}
}

string::const_iterator DnaPSSM::max_like_match(const string &target,
				float &best_logp, int &best_dir, const bool &combine_strands) const
{
	if(target.length() < m_chars.size()) {
		best_logp = NEG_INF;
		return(target.begin());
	}

	string::const_iterator max_i = target.begin() + m_max_range;
	if(max_i > target.end() - m_chars.size()) {
		max_i = target.end() - m_chars.size();
	}
	string::const_iterator best_pos;
	best_logp = NEG_INF;
	for(string::const_iterator i = target.begin() + m_min_range;
	    i <= max_i;
	    i++) {
		string::const_iterator j = i;
		float logp = 0;
		for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
		    p < m_chars.end();
		    p++) {
			if(!(*j)) {
				logp = NEG_INF;
				break;
			}
			if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
				logp += p->get_avg_log_prob();
			} else {
				int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*j];
				if(code >= 0) {
					logp += p->get_log_prob_from_code(code);
				}
			}
			j++;
		}

		float total_logp = logp;

		if(m_bidirect) {
			float rlogp = 0;
			j = i;
			for(vector<DnaProbVec>::const_reverse_iterator p = m_chars.rbegin();
			    p != m_chars.rend();
			    p++) {
				if(!(*j)) {
					rlogp = NEG_INF;
					break;
				}
				if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
					rlogp += p->get_avg_log_prob();
				} else {
					int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*j];
					if(code >= 0) {
						rlogp += p->get_log_prob_from_code(code);
					}
				}
				j++;
			}

			if (combine_strands) {
				// Combine probabilities by adding them in log space
				log_sum_log(total_logp, rlogp);
				best_dir = 0; // Indicate combined strands
			} else {
				// select best strand
				if(rlogp > logp) {
					total_logp = rlogp;
					best_dir = -1;
				} else {
					best_dir = 1;
				}
			}
		} else {
			best_dir = 1;
		}

		if(total_logp > best_logp) {
			best_logp = total_logp;
			best_pos = i;
		}
	}
	return(best_pos);
}



void DnaPSSM::integrate_like(const string &target, float &energy, vector<float> *spat_dist) const
{
	if(target.length() < m_chars.size()) {
		energy = NEG_INF;
		return;
	}

	string::const_iterator max_i = target.begin() + m_max_range;
	if(max_i > target.end() - m_chars.size()) {
		max_i = target.end() - m_chars.size();
	}
	energy = NEG_INF;
	for(string::const_iterator i = target.begin() + m_min_range;
	    i <= max_i;
	    i++) {
		string::const_iterator j = i;
		float logp = 0;
		for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
		    p < m_chars.end();
		    p++) {
			if(!(*j)) {
				logp = NEG_INF;
				break;
			}
			if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
				logp += c_log_quarter;
			} else {
				int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*j];
				if(code >= 0) {
					logp += p->get_log_prob_from_code(code);
				}
			}
			j++;
		}
   		log_sum_log(energy, logp);
		if(spat_dist) {
			log_sum_log((*spat_dist)[i - target.begin()], logp);
		}
		if(m_bidirect) {
			logp = 0;
			j = i;
			for(vector<DnaProbVec>::const_reverse_iterator
							p = m_chars.rbegin();
			    p != m_chars.rend();
			    p++) {
				if(!(*j)) {
					logp = NEG_INF;
					break;
				}
				if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
					logp += p->get_avg_log_prob();
				} else {
					int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*j];
					if(code >= 0) {
						logp += p->get_log_prob_from_code(code);
					}
				}
				j++;
			}
   			log_sum_log(energy, logp);
			if(spat_dist) {
				log_sum_log((*spat_dist)[i - target.begin()], logp);
			}
		}
	}
}

void DnaPSSM::normalize()
{
	for(vector<DnaProbVec>::iterator i = m_chars.begin();
	    i != m_chars.end();
	    i++) {
		i->normalize();
	}
}
void DnaPSSM::add_dirichlet_prior(float prior)
{
	if (prior <= 0)
		return;

	vector<float> freqs(4);
	for (vector<DnaProbVec>::iterator i = m_chars.begin();
	     i != m_chars.end();
	     ++i) {
		freqs[0] = i->get_direct_prob(0) + prior;
		freqs[1] = i->get_direct_prob(1) + prior;
		freqs[2] = i->get_direct_prob(2) + prior;
		freqs[3] = i->get_direct_prob(3) + prior;
		i->reset(freqs);
		i->normalize();
	}
}


void DnaPSSM::integrate_energy_logspat(const string &target, float &energy, vector<float> &spat_log_func, int spat_bin_size) const
{
	if(target.length() < m_chars.size()) {
		energy = NEG_INF;
		return;
	}

	string::const_iterator max_i = target.begin() + m_max_range;
	if(max_i > target.end() - m_chars.size()) {
		max_i = target.end() - m_chars.size();
	}
	energy = NEG_INF;
	int pos = 0;
	for(string::const_iterator i = target.begin() + m_min_range;
	    i <= max_i;
	    i++) {
		int spat_bin = int(pos/spat_bin_size);
		// Clamp to valid range
		if(spat_bin >= (int)spat_log_func.size()) {
			spat_bin = spat_log_func.size() - 1;
		}
		pos++;

		string::const_iterator j = i;
		float logp = 0;
		for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
		    p < m_chars.end();
		    p++) {
			if(!(*j)) {
				logp = NEG_INF;
				break;
			}
			if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
				logp += p->get_avg_log_prob();
			} else {
				int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*j];
				if(code >= 0) {
					logp += p->get_log_prob_from_code(code);
				}
			}
			j++;
		}
		logp += spat_log_func[spat_bin];
		log_sum_log(energy, logp);

		if(m_bidirect) {
			logp = 0;
			j = i;
			for(vector<DnaProbVec>::const_reverse_iterator p = m_chars.rbegin();
			    p != m_chars.rend();
			    p++) {
				if(!(*j)) {
					logp = NEG_INF;
					break;
				}
				if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
					logp += c_log_quarter;
				} else {
					int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*j];
					if(code >= 0) {
						logp += p->get_log_prob_from_code(code);
					}
				}
				j++;
			}
			logp += spat_log_func[spat_bin];
			log_sum_log(energy, logp);
		}
	}
}

void DnaPSSM::integrate_energy_max_logspat(const string &target, float &energy, vector<float> &spat_log_func, int spat_bin_size) const
{
	if(target.length() < m_chars.size()) {
		energy = NEG_INF;
		return;
	}

	string::const_iterator max_i = target.begin() + m_max_range;
	if(max_i > target.end() - m_chars.size()) {
		max_i = target.end() - m_chars.size();
	}

	energy = NEG_INF;

	int pos = 0;
	for(string::const_iterator i = target.begin() + m_min_range;
	    i <= max_i;
	    i++) {
		int spat_bin = int(pos/spat_bin_size);
		// Clamp to valid range
		if(spat_bin >= (int)spat_log_func.size()) {
			spat_bin = spat_log_func.size() - 1;
		}
		pos++;

		string::const_iterator j = i;
		float logp = 0;
		for(vector<DnaProbVec>::const_iterator p = m_chars.begin();
		    p < m_chars.end();
		    p++) {
			if(!(*j)) {
				logp = NEG_INF;
				break;
			}
			if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
				logp += p->get_avg_log_prob();
			} else {
				int code = DnaLookupTables::BASE_ENCODE[(unsigned char)*j];
				if(code >= 0) {
					logp += p->get_log_prob_from_code(code);
				}
			}
			j++;
		}
		logp += spat_log_func[spat_bin];

		if(m_bidirect) {
			float logp_rev = 0;
			j = i;
			for(vector<DnaProbVec>::const_reverse_iterator p = m_chars.rbegin();
			    p != m_chars.rend();
			    p++) {
				if(!(*j)) {
					logp_rev = NEG_INF;
					break;
				}
				if(DnaLookupTables::NEUTRAL_CHAR[(unsigned char)*j]) {
					logp_rev += c_log_quarter;
				} else {
					int code = DnaLookupTables::COMPLEMENT_ENCODE[(unsigned char)*j];
					if(code >= 0) {
						logp_rev += p->get_log_prob_from_code(code);
					}
				}
				j++;
			}
			logp_rev += spat_log_func[spat_bin];
			log_sum_log(logp, logp_rev);
		}

		if(logp > energy) {
			energy = logp;
		}
	}
}

