/*
 * PMMultitask.h
 *
 * Multitasking as R misha's IntervUtils::prepare4multitasking and distribute_task do it: the plan
 * that splits a scope among kids, and forking them. Defined in PMStubs.cpp but for the templates.
 */

#ifndef PMMULTITASK_H_
#define PMMULTITASK_H_

#include <cstdint>
#include <new>
#include <vector>

#include "pymisha.h"
#include "GInterval.h"
#include "TGLException.h"

// An interval of a kid's share of the scope, and its row in the scope (0-based)
struct SplitInterval {
    GInterval interval;
    uint64_t original_idx;
};

// A kid's intervals and, if id_remap, their rows in the scope
void extract_sub_intervals(const std::vector<SplitInterval> &split, std::vector<GInterval> &sub,
                           std::vector<uint64_t> *id_remap = nullptr);

// The scope of each kid of a multitasking call, as R misha's IntervUtils::prepare4multitasking
// plans it, for at most min(2 x the cores, max_processes) kids (2 is R's gmax.processes2core).
// range_split_bin > 0 (a fixed-bin iterator's bin size, where R splits ranges): ranges of about the
// same size cut at bin boundaries, at most one kid per max(min_scope4process, 500 kb) and per 50000
// bins. Otherwise whole chromosomes: a kid starts a new one at a chromosome once it holds more
// than min_scope4process bp and, with that chromosome, its share of the rest. The kids' intervals
// are consecutive runs of the scope, so the kids' rows in kid order are the rows one process
// gives. iterator_policy: a bin size or an iterator of intervals (-1) scans the scope in its order;
// the implicit iterator (0) of a sparse or array track reads it a chromosome at a time instead.
// Empty when the call runs in this process: multitasking is off, the plan has fewer than two
// kids, or the scope's chromosomes are out of order with the implicit iterator, which would give
// the rows in another order (a range split keeps a bin iterator's order).
std::vector<std::vector<SplitInterval>> plan_kids(const std::vector<GInterval> &intervals, int64_t range_split_bin,
                                                  int64_t iterator_policy);

// Runs work() in a forked kid and ends the kid: _exit(0), or _exit(1) once the error is handed to the
// parent. No exception leaves it for the caller's frames, whose destructors would call into Python.
template <typename F>
[[noreturn]] void run_kid(F &&work)
{
    try {
        work();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
    } catch (...) {
        PyMisha::handle_error("Unknown error in a multitasking process");
    }
    _exit(0);
}

// Forks a kid per element of plan and waits for them all. Kid k runs
// kid_work(k, intervals, ids): its intervals and, per interval, its row in the scope (0-based).
// PyMisha::prepare4multitasking() with the result sizes comes first.
template <typename F>
void fork_kids(const std::vector<std::vector<SplitInterval>> &plan, F &&kid_work)
{
    for (int kid = 0; kid < (int)plan.size(); ++kid) {
        if (!PyMisha::launch_process()) {
            run_kid([&] {
                std::vector<GInterval> sub;
                std::vector<uint64_t> ids;
                extract_sub_intervals(plan[kid], sub, &ids);
                kid_work(kid, sub, ids);
            });
        }
    }
    while (PyMisha::wait_for_kids(1))
        ;
}

#endif /* PMMULTITASK_H_ */
