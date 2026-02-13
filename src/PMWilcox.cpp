// pm_segment and pm_wilcox implementations
// Ported from R misha GenomeTrackSegmentation.cpp and GenomeTrackWilcox.cpp

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDb.h"
#include "PMTrackExpressionScanner.h"
#include "PMTrackExpressionIterator.h"
#include "IncrementalWilcox.h"

#include <new>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

// Forward declarations of helpers from PMStubs.cpp
extern void convert_py_intervals(PyObject *py_intervals, std::vector<GInterval> &intervals);
extern long parse_iterator_policy(PyObject *py_iterator, long default_policy, const char *caller);

// ================================= gwilcox =================================

// Sliding-window Wilcoxon test result: interval + minimum p-value
struct IntervalPval {
    int32_t chromid;
    int64_t start;
    int64_t end;
    double minpval;

    IntervalPval(int32_t c, int64_t s, int64_t e, double p)
        : chromid(c), start(s), end(e), minpval(p) {}
};

class SlidingWilcox {
public:
    enum What2find { FIND_LOWS_AND_HIGHS, FIND_LOWS, FIND_HIGHS };

private:
    enum { LARGE, SMALL, NUM_WINS };

    What2find m_what2find;
    unsigned m_num_read_samples;
    unsigned m_winsize[NUM_WINS];
    unsigned m_winsize_aside[NUM_WINS];
    unsigned m_tail[NUM_WINS];
    unsigned m_center;
    std::vector<double> m_queue;

    unsigned          m_binsize;
    int64_t           m_start_coord;
    int64_t           m_center_coord;
    double            m_maxz;
    double            m_minpval;
    int32_t           m_chromid;
    std::vector<IntervalPval> &m_intervals;

    IncrementalWilcox m_wilcox;

public:
    SlidingWilcox(bool one_tailed, What2find what2find,
                  unsigned winsize_in_coord1, unsigned winsize_in_coord2,
                  unsigned binsize, int32_t chromid,
                  std::vector<IntervalPval> &res_intervals, double maxz);

    ~SlidingWilcox();

    void set_next_sample(double v);
};

SlidingWilcox::SlidingWilcox(bool one_tailed, What2find what2find,
                             unsigned winsize_in_coord1, unsigned winsize_in_coord2,
                             unsigned binsize, int32_t chromid,
                             std::vector<IntervalPval> &res_intervals, double maxz) :
    m_what2find(what2find), m_num_read_samples(0), m_binsize(binsize),
    m_start_coord(-1), m_center_coord(0), m_maxz(maxz), m_minpval(0),
    m_chromid(chromid), m_intervals(res_intervals), m_wilcox(one_tailed)
{
    unsigned winsize_in_coord[NUM_WINS];

    winsize_in_coord[LARGE] = std::max(winsize_in_coord1, winsize_in_coord2);
    winsize_in_coord[SMALL] = std::min(winsize_in_coord1, winsize_in_coord2);

    for (int i = 0; i < NUM_WINS; i++) {
        m_winsize_aside[i] = (unsigned)(0.5 * winsize_in_coord[i] / m_binsize + 0.5);
        m_winsize[i] = 2 * m_winsize_aside[i] + 1;
        if (m_winsize[i] < IncrementalWilcox::MIN_RELIABLE_WINSIZE)
            verror("Window of size %d contains too few samples (%d) to run Wilcoxon test",
                   winsize_in_coord[i], m_winsize[i]);
    }

    m_queue.resize(m_winsize[LARGE], std::numeric_limits<double>::quiet_NaN());

    m_center = m_winsize_aside[LARGE];
    m_tail[LARGE] = 0;
    m_tail[SMALL] = (m_winsize_aside[LARGE] + m_winsize_aside[SMALL] + 1) % m_winsize[LARGE];

    m_start_coord = -1;
    m_center_coord = -1 * int64_t(m_binsize * m_winsize_aside[LARGE]);
}

SlidingWilcox::~SlidingWilcox()
{
    for (unsigned i = 0; i <= m_winsize_aside[LARGE]; i++)
        set_next_sample(std::numeric_limits<double>::quiet_NaN());
}

void SlidingWilcox::set_next_sample(double v)
{
    m_num_read_samples++;

    double old_v[NUM_WINS];
    double new_v[NUM_WINS];

    old_v[LARGE] = m_queue[m_tail[LARGE]];
    old_v[SMALL] = m_queue[(m_tail[SMALL] + m_winsize[LARGE] - m_winsize[SMALL]) % m_winsize[LARGE]];

    m_queue[m_tail[LARGE]] = v;

    new_v[LARGE] = v;
    new_v[SMALL] = m_queue[m_tail[SMALL]];

    m_tail[LARGE] = (m_tail[LARGE] + 1) % m_winsize[LARGE];
    m_tail[SMALL] = (m_tail[SMALL] + 1) % m_winsize[LARGE];
    m_center = (m_center + 1) % m_winsize[LARGE];

    m_wilcox.update(old_v[LARGE], new_v[LARGE], old_v[SMALL], new_v[SMALL]);
    double z;

    if (m_what2find == FIND_HIGHS)
        z = m_wilcox.z_highs();
    else if (m_what2find == FIND_LOWS)
        z = m_wilcox.z_lows();
    else
        z = m_wilcox.z();

    if (std::isnan(m_queue[m_center]) || z > m_maxz) {
        if (m_start_coord != -1) {
            int64_t start = std::max((int64_t)0, m_start_coord - m_winsize_aside[SMALL] * m_binsize);
            int64_t end = m_center_coord + m_winsize_aside[SMALL] * m_binsize;

            if (m_intervals.empty() || m_intervals.back().chromid != m_chromid || m_intervals.back().end < start)
                m_intervals.push_back(IntervalPval(m_chromid, start, end, m_minpval));
            else {
                m_intervals.back().end = end;
                m_intervals.back().minpval = std::min(m_intervals.back().minpval, m_minpval);
            }
            m_start_coord = -1;
        }
    } else {
        double pval;

        if (m_what2find == FIND_HIGHS)
            pval = m_wilcox.pval_highs();
        else if (m_what2find == FIND_LOWS)
            pval = m_wilcox.pval_lows();
        else
            pval = m_wilcox.pval();

        if (m_start_coord == -1) {
            m_start_coord = m_center_coord;
            m_minpval = pval;
        } else
            m_minpval = std::min(m_minpval, pval);
    }

    m_center_coord += m_binsize;
}

// pm_wilcox(expr, intervals, winsize1, winsize2, maxpval, onetailed, what2find, iterator, config)
PyObject *pm_wilcox(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_intervals = NULL;
        double winsize1 = 0;
        double winsize2 = 0;
        double maxpval = 0.05;
        int onetailed = 1;
        int what2find = 1;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOdddii|OO",
                              &py_expr, &py_intervals,
                              &winsize1, &winsize2, &maxpval,
                              &onetailed, &what2find,
                              &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_wilcox");
        }

        if (!PyUnicode_Check(py_expr))
            verror("gwilcox expression must be a string");
        std::string expr = PyUnicode_AsUTF8(py_expr);

        if (winsize1 < 0 || winsize2 < 0)
            verror("Winsize cannot be a negative number");
        if (winsize1 != (int)winsize1 || winsize2 != (int)winsize2)
            verror("Winsize must be an integer");

        // Convert maxpval to z-score (done in Python layer via scipy.stats.norm.ppf)
        // Here maxpval is already the z-score passed from Python
        double maxz = maxpval;  // Python sends pre-computed z-score

        SlidingWilcox::What2find w2f;
        if (what2find < 0)
            w2f = SlidingWilcox::FIND_LOWS;
        else if (what2find > 0)
            w2f = SlidingWilcox::FIND_HIGHS;
        else
            w2f = SlidingWilcox::FIND_LOWS_AND_HIGHS;

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_wilcox");

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Create scanner
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        // Verify fixed-bin iterator
        PMFixedBinIterator *fbi = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fbi)
            verror("gwilcox() requires the iterator policy to be a fixed bin size");

        unsigned bin_size = (unsigned)fbi->get_bin_size();

        std::vector<IntervalPval> res_intervals;
        SlidingWilcox *wilcox = NULL;
        GInterval last_interval(-1, -1, -1);

        for (; !scanner.isend(); scanner.next()) {
            const GInterval &cur_interval = scanner.last_interval();

            if (last_interval.chromid != cur_interval.chromid || last_interval.end != cur_interval.start) {
                delete wilcox;
                wilcox = NULL;
                wilcox = new SlidingWilcox(onetailed != 0, w2f,
                                           (unsigned)winsize1, (unsigned)winsize2,
                                           bin_size, cur_interval.chromid,
                                           res_intervals, maxz);
            }
            wilcox->set_next_sample(scanner.vdouble(0));
            g_pymisha->verify_max_data_size(res_intervals.size(), "Result");
            last_interval = cur_interval;

            check_interrupt();
        }
        delete wilcox;

        if (res_intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Build result DataFrame: chrom, start, end, pval
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        PMDataFrame df(res_intervals.size(), 4, "intervals_pval");
        df.init_col(0, "chrom", PMDataFrame::STR);
        df.init_col(1, "start", PMDataFrame::LONG);
        df.init_col(2, "end", PMDataFrame::LONG);
        df.init_col(3, "pval", PMDataFrame::DOUBLE);

        for (size_t i = 0; i < res_intervals.size(); ++i) {
            df.val_str(i, 0, chromkey.id2chrom(res_intervals[i].chromid).c_str());
            df.val_long(i, 1, res_intervals[i].start);
            df.val_long(i, 2, res_intervals[i].end);
            df.val_double(i, 3, res_intervals[i].minpval);
        }

        PMPY result = df.construct_py(true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}


// ================================= gsegment =================================

struct Winval {
    double  v;
    int64_t coord;

    Winval(double _v, int64_t _coord) : v(_v), coord(_coord) {}
};

// pm_segment(expr, intervals, minsegment, maxpval, onetailed, iterator, config)
PyObject *pm_segment(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_intervals = NULL;
        double minsegment = 0;
        double maxpval = 0.05;
        int onetailed = 1;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOddi|OO",
                              &py_expr, &py_intervals,
                              &minsegment, &maxpval,
                              &onetailed,
                              &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_segment");
        }

        if (!PyUnicode_Check(py_expr))
            verror("gsegment expression must be a string");
        std::string expr = PyUnicode_AsUTF8(py_expr);

        if (minsegment < 0)
            verror("Min segment cannot be a negative number");
        if (minsegment != (int)minsegment)
            verror("Min segment must be an integer");

        // maxpval is already converted to z-score in Python layer
        double maxz = maxpval;

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_segment");

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Create scanner
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        // Verify fixed-bin iterator
        PMFixedBinIterator *fbi = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fbi)
            verror("gsegment() requires the iterator policy to be a fixed bin size");

        unsigned bin_size = (unsigned)fbi->get_bin_size();
        unsigned winsize = std::max((unsigned)(minsegment / bin_size + 0.5), 1u);

        std::vector<GInterval> segments;
        std::vector<Winval> window;
        std::vector<double> segment_tail;

        int cur_scope_idx = -1;
        IncrementalWilcox wilcox(onetailed != 0);
        int32_t cur_chromid = -1;
        int64_t cur_coord = 0;
        int64_t segment_start_coord = 0;
        int64_t segment_end_coord = 0;
        int64_t best_end_coord = 0;
        bool segment_closing = false;
        double best_z = 1;
        unsigned best_sample_idx = 0;
        unsigned winidx = 0;

        // We track scope_idx by detecting discontinuities
        int scope_idx = 0;
        GInterval prev_interval(-1, -1, -1);

        for (; ; scanner.next()) {
            if (scanner.isend() || prev_interval.chromid != scanner.last_interval().chromid ||
                (!scanner.isend() && prev_interval.end != scanner.last_interval().start)) {

                if (cur_scope_idx >= 0) {
                    // Interval ended - extend the last segment to cover remaining values
                    if (segments.empty() || segments.back().chromid != cur_chromid) {
                        GInterval interval(cur_chromid, segment_start_coord, cur_coord);
                        segments.push_back(interval);
                        g_pymisha->verify_max_data_size(segments.size(), "Result");
                    } else {
                        segments.back().end = cur_coord;
                    }
                }

                if (scanner.isend())
                    break;

                cur_scope_idx = scope_idx++;
                cur_chromid = scanner.last_interval().chromid;
                wilcox.reset();
                cur_coord = segment_start_coord = segment_end_coord = best_end_coord = scanner.last_interval().start;
                segment_closing = false;
                best_z = 1;
                best_sample_idx = 0;
                winidx = 0;
                window.clear();
                segment_tail.clear();
                window.resize(winsize, Winval(-1, 0));
                segment_tail.resize(winsize, -1);
            }

            double v = scanner.vdouble(0);
            prev_interval = scanner.last_interval();

            if (!std::isnan(v)) {
                // 1. fill first the current segment
                if (wilcox.n1() < winsize) {
                    wilcox.update(std::numeric_limits<double>::quiet_NaN(), v,
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
                    segment_end_coord = cur_coord;
                }

                // 2. then fill the window that comes afterwards
                else if (wilcox.n2() < winsize) {
                    wilcox.update(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(), v);
                    window[winidx] = Winval(v, cur_coord);
                    winidx = (winidx + 1) % winsize;

                // 3. now increase the segment and move the window
                } else {
                    double old_v = window[winidx].v;
                    wilcox.update(std::numeric_limits<double>::quiet_NaN(), old_v, old_v, v);
                    segment_end_coord = window[winidx].coord;
                    window[winidx] = Winval(v, cur_coord);
                    winidx = (winidx + 1) % winsize;
                    if (segment_closing)
                        segment_tail.push_back(old_v);
                }

                if (wilcox.n2() >= winsize) {
                    double z = wilcox.z();

                    if (z > 0)
                        verror("Wilcoxon test is unreliable on small windows. Min segment size must be at least %d",
                               IncrementalWilcox::MIN_RELIABLE_WINSIZE * bin_size);

                    if (segment_closing) {
                        if (z < best_z) {
                            best_z = z;
                            best_end_coord = segment_end_coord;
                            best_sample_idx = segment_tail.size();
                        }

                        // at the end of buffer intended for checking => close the segment
                        if (segment_tail.size() >= winsize - 1) {
                            best_end_coord += bin_size;
                            GInterval interval(cur_chromid, segment_start_coord, best_end_coord);
                            segments.push_back(interval);
                            g_pymisha->verify_max_data_size(segments.size(), "Result");
                            segment_closing = false;
                            segment_start_coord = best_end_coord;

                            // Reinitiate Wilcoxon algorithm
                            wilcox.reset();

                            unsigned num_tail_samples = (unsigned)(winsize - best_sample_idx - 1);
                            for (unsigned i = segment_tail.size() - num_tail_samples; i < segment_tail.size(); i++)
                                wilcox.update(std::numeric_limits<double>::quiet_NaN(), segment_tail[i],
                                              std::numeric_limits<double>::quiet_NaN(),
                                              std::numeric_limits<double>::quiet_NaN());
                            segment_tail.clear();

                            for (unsigned i = 0; i < winsize - num_tail_samples; i++) {
                                wilcox.update(std::numeric_limits<double>::quiet_NaN(), window[winidx].v,
                                              std::numeric_limits<double>::quiet_NaN(),
                                              std::numeric_limits<double>::quiet_NaN());
                                segment_end_coord = window[winidx].coord;
                                winidx = (winidx + 1) % winsize;
                            }

                            for (unsigned i = 0; i < num_tail_samples; i++) {
                                wilcox.update(std::numeric_limits<double>::quiet_NaN(),
                                              std::numeric_limits<double>::quiet_NaN(),
                                              std::numeric_limits<double>::quiet_NaN(), window[winidx].v);
                                winidx = (winidx + 1) % winsize;
                            }
                        }

                    // segment is not closing yet
                    } else if (z <= maxz) {
                        segment_closing = true;
                        best_z = z;
                        best_end_coord = segment_end_coord;
                        best_sample_idx = 0;
                        segment_tail.clear();
                    }
                }
            }

            cur_coord = scanner.last_interval().end;

            check_interrupt();
        }

        if (segments.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Build result DataFrame: chrom, start, end
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        PMDataFrame df(segments.size(), 3, "intervals");
        df.init_col(0, "chrom", PMDataFrame::STR);
        df.init_col(1, "start", PMDataFrame::LONG);
        df.init_col(2, "end", PMDataFrame::LONG);

        for (size_t i = 0; i < segments.size(); ++i) {
            df.val_str(i, 0, chromkey.id2chrom(segments[i].chromid).c_str());
            df.val_long(i, 1, segments[i].start);
            df.val_long(i, 2, segments[i].end);
        }

        PMPY result = df.construct_py(true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}
