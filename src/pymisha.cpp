#include <time.h>

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdarg.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "pymisha.h"
#include "pmutils.h"

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

using namespace std;

// Semaphore locker helper class
class SemLocker {
public:
    SemLocker(sem_t *sem) : m_sem(sem) { sem_wait(m_sem); }
    ~SemLocker() { sem_post(m_sem); }
private:
    sem_t *m_sem;
};

struct sigaction         PyMisha::s_old_sigint_act;
struct sigaction         PyMisha::s_old_sigalrm_act;
struct sigaction         PyMisha::s_old_sigchld_act;
int                      PyMisha::s_ref_count = 0;
int                      PyMisha::s_sigint_fired = 0;
bool                     PyMisha::s_sigalrm_fired = false;
bool                     PyMisha::s_is_kid = false;
pid_t                    PyMisha::s_parent_pid = 0;
sem_t                   *PyMisha::s_shm_sem = SEM_FAILED;
int                      PyMisha::s_kid_index;
vector<pid_t>            PyMisha::s_running_pids;
PyMisha::Shm            *PyMisha::s_shm = (PyMisha::Shm *)MAP_FAILED;
char                   *PyMisha::s_res = (char *)MAP_FAILED;
uint64_t                 PyMisha::s_res_size = 0;
uint64_t                 PyMisha::s_res_const_size = 0;
uint64_t                 PyMisha::s_res_record_size = 0;
unsigned                 PyMisha::s_res_num_results = 1;
bool                    PyMisha::s_res_size_from_ram = false;
bool                     PyMisha::s_multitasking_stdout = false;

PyMisha *g_pymisha = NULL;
unsigned g_transact_id = 0;
string PyMisha::s_groot;
string PyMisha::s_uroot;
extern PyObject *s_pm_err;

PyMisha::PyMisha(bool check_db)
{
    // Checked FIRST, before anything is initialised and before s_ref_count++.
    // C++ does not run the destructor of an object whose constructor threw, so a
    // throw from any later point leaks whatever this constructor has already set:
    // the ref count (permanently skipping the init block for every later call,
    // leaving g_pymisha pointing at a dead stack frame), the umask, and the
    // TGLException error handler.
    if (check_db && s_groot.empty())
        verror("Database was not loaded. Please call gdb.init.");

    if (!s_ref_count) {
        // The three pieces of global state the destructor restores are taken first,
        // so the catch below has something valid to put back.
        m_old_umask = umask(07);
        m_old_error_handler = TGLException::set_error_handler(TGLException::throw_error_handler);
        g_pymisha = this;

        // Everything from here on can throw - verror() does, and so can
        // load_options(). ~PyMisha() does not run for an object whose constructor
        // threw, so unwind by hand and rethrow.
        try {

        ++g_transact_id;

        // the first random draw of this call seeds from numpy's generator (pm::rng_load), unless
        // the call continues the stream of the calls before it (pm::g_rng_stream)
        if (!pm::g_rng_stream)
            pm::g_rng_loaded = false;
        pm::g_rng_peeked = false;

        s_sigint_fired = 0;
        s_sigalrm_fired = 0;
        s_is_kid = false;
        s_kid_index = 0;
        s_parent_pid = getpid();
        s_shm_sem = SEM_FAILED;
        s_shm = (Shm *)MAP_FAILED;
        s_res = (char *)MAP_FAILED;
        s_res_size = s_res_const_size = s_res_record_size = 0;
        s_running_pids.clear();

        PMPY module(PyImport_AddModule("__main__"));
        if (!module)
            verror("Failed to access __main__ module");

        m_gdict.assign(PyModule_GetDict(module));
        if (!m_gdict)
            verror("Failed to access global dictionary");

        m_c_module_dict.assign(PyModule_GetDict(g_module));
        if (!m_c_module_dict)
            verror("Failed to access C module dictionary");

        m_py_module_dict.assign(PyDict_GetItemString(m_gdict, "_PM_REGRESSION_TEST_DICT"));
        if (!m_py_module_dict)
            m_py_module_dict.assign(PyDict_GetItemString(m_c_module_dict, "_PMLOCALS"));
        // If not found yet, try importing the pymisha module and getting its _PMLOCALS
        if (!m_py_module_dict) {
            PMPY pymisha_module(PyImport_ImportModule("pymisha"), true);
            if (pymisha_module) {
                PMPY pymisha_dict(PyModule_GetDict(pymisha_module), false);
                if (pymisha_dict) {
                    m_py_module_dict.assign(PyDict_GetItemString(pymisha_dict, "_PMLOCALS"));
                }
            }
        }
        // If still not found, just use an empty dict for now
        if (!m_py_module_dict) {
            m_py_module_dict.assign(PyDict_New(), true);
            vdebug("Warning: Using empty module dictionary\n");
        }

        load_options();

        } catch (...) {
            TGLException::set_error_handler(m_old_error_handler);
            g_pymisha = NULL;
            umask(m_old_umask);
            throw;
        }
    }

    s_ref_count++;
}

PyMisha::~PyMisha()
{
    s_ref_count--;

    if (!s_ref_count) {
        // if this is a child, do not detach from shared memory and do not deallocate the semaphore
        if (!s_is_kid) {
            check_kids_state(true);

            if (s_shm_sem != SEM_FAILED) {
                SemLocker sl(s_shm_sem);
                SigBlocker sb;

                // kill all the remaining child processes
                for (vector<pid_t>::const_iterator ipid = s_running_pids.begin(); ipid != s_running_pids.end(); ++ipid) {
                    vdebug("Forcefully terminating process %d\n", *ipid);
                    kill(*ipid, SIGTERM);
                }
            }

            // After SIGTERM is sent to all the kids, wait till they are reaped. A kid that another
            // waitpid() reaped is no longer ours and is dropped (check_kids_state), or the wait
            // would not end. A kid still alive after 10 s gets SIGKILL; after 30 s the rest are
            // given up on (as R misha 5.17).
            struct timespec timeout;
            for (int waited_ms = 0;; waited_ms += 10) {
                check_kids_state(true);
                if (s_running_pids.empty())
                    break;

                vdebug("Waiting for %ld child processes to end\n", s_running_pids.size());
                if (waited_ms == 10000) {
                    for (vector<pid_t>::const_iterator ipid = s_running_pids.begin(); ipid != s_running_pids.end(); ++ipid)
                        kill(*ipid, SIGKILL);
                }
                if (waited_ms >= 30000) {
                    s_running_pids.clear();
                    break;
                }

                timeout.tv_sec = 0;
                timeout.tv_nsec = 10000000; // 10ms
                nanosleep(&timeout, NULL);
            }

            if (s_shm_sem != SEM_FAILED)
                sem_close(s_shm_sem);

            if (s_shm != (Shm *)MAP_FAILED)
                munmap(s_shm, sizeof(Shm));

            if (s_res != (char *)MAP_FAILED)
                munmap(s_res, s_res_size);
        }

        TGLException::set_error_handler(m_old_error_handler);

        // reset alarm
        alarm(0);

        umask(m_old_umask);
    }

    if (!s_ref_count)
        g_pymisha = NULL;
}

string PyMisha::get_shm_sem_name()
{
    char buf[100];
    snprintf(buf, sizeof(buf), "pymisha-shm-sem-%d", (int)getpid());
    return buf;
}

// A fork copies the page tables of the parent's anonymous memory, one entry per page: the
// cost grows with the memory the session holds in 4 KB pages, and is paid once per kid.
// Memory in 2 MB transparent huge pages costs 512 times fewer entries. Measured by R misha on
// CentOS 7 (Linux 3.10, no huge pages for the R heap) with 32 kids: about 0.035 s per GB per
// kid; on AlmaLinux 8, the same 20 GB in huge pages forked in no time.
//
// The warning comes when the forks of this call would take about 4.5 s or more: GB in 4 KB
// pages times kids of 128 or more (4 GB with 32 kids, 8 GB with 16, 32 GB with 4), once per
// session, as in R misha (rdb::warn_if_forks_are_slow).
//
// The check is cheap on every call: the anonymous resident size comes from /proc/self/statm.
// Only when that alone is large enough does it read how much of it is in huge pages
// (/proc/self/smaps_rollup, or /proc/self/smaps before Linux 4.14, which walk the page tables
// of the session). That is read again only when the anonymous memory has grown or shrunk by a
// quarter since; in between the bytes in 4 KB pages found then (capped by the anonymous size)
// are multiplied by each call's number of kids.
//
// MISHA_TEST_FORK_4K_BYTES, an environment variable for the tests, stands in for the memory:
// "A" for A bytes of anonymous memory in 4 KB pages, "A,H" for A bytes of which H are in huge
// pages.
static const double FORK_COST_GB_KIDS = 128;
static bool s_fork_cost_warned = false;
static uint64_t s_fork_cost_checked_anon = 0;
static uint64_t s_fork_cost_small_pages = 0;

// Anonymous resident size of the process in bytes: resident minus file-backed and shared
// memory, from /proc/self/statm. 0 on failure (or where there is no /proc).
static uint64_t get_anon_resident_mem_usage()
{
    FILE *fp = fopen("/proc/self/statm", "r");
    if (!fp)
        return 0;

    unsigned long size = 0, resident = 0, shared = 0;
    int n = fscanf(fp, "%lu %lu %lu", &size, &resident, &shared);
    fclose(fp);

    long page_size = sysconf(_SC_PAGESIZE);
    if (n != 3 || page_size <= 0 || resident < shared)
        return 0;
    return (uint64_t)(resident - shared) * (uint64_t)page_size;
}

// Bytes of the process's anonymous memory in transparent huge pages: AnonHugePages of
// /proc/self/smaps_rollup, or the sum over /proc/self/smaps where there is no smaps_rollup.
// Both walk the page tables, so it is called rarely. 0 on failure.
static uint64_t get_anon_huge_pages()
{
    char line[256];
    uint64_t kb = 0;

    FILE *fp = fopen("/proc/self/smaps_rollup", "r");
    if (!fp) {
        fp = fopen("/proc/self/smaps", "r");
        if (!fp)
            return 0;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (!strncmp(line, "AnonHugePages:", 14))
            kb += strtoull(line + 14, NULL, 10);
    }
    fclose(fp);
    return kb * 1024;
}

static void warn_if_forks_are_slow(unsigned num_kids)
{
    if (s_fork_cost_warned || PyMisha::is_kid() || num_kids < 2)
        return;

    const double GB = 1e9;
    const char *test_memory = getenv("MISHA_TEST_FORK_4K_BYTES");
    uint64_t test_anon = 0, test_huge = 0;

    if (test_memory) {
        char *end = NULL;
        test_anon = strtoull(test_memory, &end, 10);
        if (end && *end == ',')
            test_huge = strtoull(end + 1, NULL, 10);
    }

    uint64_t anon = test_memory ? test_anon : get_anon_resident_mem_usage();

    if (anon / GB * num_kids < FORK_COST_GB_KIDS)
        return;

    if (!s_fork_cost_checked_anon || anon >= s_fork_cost_checked_anon + s_fork_cost_checked_anon / 4 ||
        anon <= s_fork_cost_checked_anon - s_fork_cost_checked_anon / 4)
    {
        uint64_t huge = test_memory ? test_huge : get_anon_huge_pages();

        s_fork_cost_small_pages = anon > huge ? anon - huge : 0;
        s_fork_cost_checked_anon = anon;
    }

    uint64_t small_pages = std::min(s_fork_cost_small_pages, anon);

    if (small_pages / GB * num_kids < FORK_COST_GB_KIDS)
        return;

    char msg[1000];
    snprintf(msg, sizeof(msg),
             "This Python session holds %.1f GB of memory in 4 KB pages (not in transparent huge pages). "
             "Each of the %u processes pymisha starts for this call is a fork of the session, and every fork "
             "copies the page tables of that memory: this can add seconds, up to tens of seconds, to every "
             "multitasking call, however small. Lower CONFIG['max_processes'], free large objects (del, then "
             "gc.collect()), or run on a host whose kernel keeps the session's memory in transparent huge pages "
             "(such as RHEL 8 and its clones). This warning is shown once per session.",
             small_pages / GB, num_kids);

    // Python shows a warning when it is raised, so it counts as shown from here
    s_fork_cost_warned = true;
    if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) {
        // warnings turned into errors
        PyErr_Clear();
        verror("%s", msg);
    }
}

void PyMisha::prepare4multitasking(unsigned num_kids)
{
    warn_if_forks_are_slow(num_kids);
    s_res_record_size = 0;   // no result arena (the overload below maps one)

    // the seed of the kids' random draws, which they do not take from Python themselves
    if (!pm::g_rng_loaded && !pm::g_rng_peeked)
        pm::rng_peek();

    vdebug("Cleaning old semaphores\n");
    if (s_shm_sem == SEM_FAILED) {
        sem_unlink(get_shm_sem_name().c_str());
        if ((s_shm_sem = sem_open(get_shm_sem_name().c_str(), O_CREAT | O_EXCL, 0644, 1)) == SEM_FAILED)
            verror("sem_open failed: %s", strerror(errno));

        sem_unlink(get_shm_sem_name().c_str());
    }

    vdebug("Allocating shared memory for internal communication\n");
    if (s_shm == (Shm *)MAP_FAILED) {
        s_shm = (Shm *)mmap(NULL, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

        if (s_shm == (Shm *)MAP_FAILED)
            verror("Failed to allocate shared memory: %s", strerror(errno));

        s_shm->error_msg[0] = '\0';
        for (int i = 0; i < MAX_KIDS; ++i)
            s_shm->kid_progress[i] = 0;
    }
}

// The smallest arena ram_shm_size() gives, whatever CONFIG['max_mem_usage'] is (R misha's
// MIN_RAM_SHM_SIZE)
static const uint64_t MIN_RAM_SHM_SIZE = (uint64_t)16 << 20;

uint64_t PyMisha::ram_shm_size()
{
    // Half the RAM: once the kids are done the parent copies their results out of the arena into
    // arrays of about the same size, so arena and copy together fit in the RAM
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    uint64_t ram = pages > 0 && page_size > 0 ? (uint64_t)pages * (uint64_t)page_size : (uint64_t)8e9;
    return max(min(ram / 2, g_pymisha->max_mem_usage()), MIN_RAM_SHM_SIZE);
}

uint64_t PyMisha::kid_buf_size(unsigned num_kids, uint64_t record_size)
{
    if (g_pymisha->max_data_size() < MAX_DATA_SIZE_NO_LIMIT)
        return (uint64_t)ceil(g_pymisha->max_data_size() / (double)num_kids);
    return max(ram_shm_size() / ((uint64_t)max(num_kids, 1u) * record_size), (uint64_t)1);
}

bool PyMisha::prepare4multitasking(unsigned num_kids, uint64_t res_const_size, uint64_t res_record_size,
                                   uint64_t max_records, unsigned num_results)
{
    if (!s_running_pids.empty())
        verror("Multitasking: the kids of a previous step are still running");

    if (s_res != (char *)MAP_FAILED) {
        munmap(s_res, s_res_size);
        s_res = (char *)MAP_FAILED;
    }

    // Sized for the most the limit allows; the pages are committed only as the kids write.
    // Saturated, so that a large limit cannot wrap the size around.
    const uint64_t cap = (uint64_t)1 << 62;
    uint64_t limit;
    if (__builtin_mul_overflow(g_pymisha->max_data_size(), (uint64_t)num_results, &limit))
        limit = cap;
    if (max_records && max_records < limit)
        limit = max_records;
    uint64_t const_size, size;
    if (__builtin_mul_overflow((uint64_t)num_kids, res_const_size + 8, &const_size))
        const_size = cap;
    if (__builtin_mul_overflow(limit, res_record_size, &size) || __builtin_add_overflow(size, const_size, &size) ||
        size > cap)
        size = cap;
    size = max(size, (uint64_t)8);

    s_res_size_from_ram = false;
    s_res = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (s_res == (char *)MAP_FAILED) {
        // No limit, or a limit larger than can be mapped: map one sized from the RAM instead, if
        // the kids' fixed-size parts fit in it (as R misha). A result that exceeds it fails in
        // allocate_res().
        uint64_t ram_size = ram_shm_size();
        if (size > ram_size && const_size <= ram_size) {
            size = ram_size;
            s_res_size_from_ram = true;
            s_res = (char *)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        }
        // Not even that: the caller runs in one process. R misha reports an error here, except in
        // gextract and gscreen, which then run in one process too.
        if (s_res == (char *)MAP_FAILED) {
            vdebug("Failed to allocate shared memory for %u kids: %s\n", num_kids, strerror(errno));
            return false;
        }
    }
    s_res_size = size;

    prepare4multitasking(num_kids);

    // Kid i of this step is s_kid_index i, and its result goes into slot i
    s_kid_index = 0;
    s_shm->num_kids = num_kids;
    s_shm->res_offset = 0;
    for (int i = 0; i < MAX_KIDS; ++i) {
        s_shm->kid_res_offset[i] = NO_RES;
        s_shm->kid_res_num_records[i] = 0;
        s_shm->kid_progress[i] = 0;
    }
    s_res_const_size = res_const_size;
    s_res_record_size = res_record_size;
    s_res_num_results = num_results;
    return true;
}

void PyMisha::verify_kids_records(uint64_t num_records)
{
    // Over num_results results, the largest holds at least num_records / num_results records, more
    // than the limit when num_records is more than num_results times it
    g_pymisha->verify_max_data_size((num_records + s_res_num_results - 1) / s_res_num_results, "Result");
}

void *PyMisha::allocate_res(uint64_t num_records)
{
    if (!s_is_kid)
        verror("allocate_res() cannot be called by the parent process");

    s_shm->kid_res_num_records[s_kid_index] = num_records;

    uint64_t res_size;
    bool overflow = __builtin_mul_overflow(num_records, s_res_record_size, &res_size) ||
                    __builtin_add_overflow(res_size, s_res_const_size + 7, &res_size);
    res_size &= ~(uint64_t)7;   // rounded up to 8 bytes, so that every kid's result starts aligned

    uint64_t offset = 0;
    {
        SemLocker sl(s_shm_sem);
        if (!overflow) {
            offset = s_shm->res_offset;
            overflow = res_size > s_res_size - min(offset, s_res_size);
            if (!overflow) {
                s_shm->kid_res_offset[s_kid_index] = offset;
                s_shm->res_offset = offset + res_size;
            }
        }
    }

    // reported once the lock is released: a kid's error takes it too
    if (overflow) {
        // an arena sized by the limit overflows when the kids' records exceed it: the error the
        // parent gives for that, whoever comes first
        if (!s_res_size_from_ram) {
            uint64_t total = 0;
            for (unsigned i = 0; i < s_shm->num_kids; ++i)
                total += s_shm->kid_res_num_records[i];
            verify_kids_records(total);
        }
        if (s_res_size_from_ram)
            verror("Result size exceeded the maximal allowed.\n"
                   "Try to bound the scope of the function.\n"
                   "Note: CONFIG['max_data_size'] sets no limit or more than can be mapped, so the result had to fit\n"
                   "into half the RAM, at most CONFIG['max_mem_usage'].");
        verror("Result size exceeded the maximal allowed.\n"
               "Try to bound the scope of the function.\n"
               "Note: the maximum data size is controlled via CONFIG['max_data_size'].");
    }
    return s_res + offset;
}

void *PyMisha::get_kid_res(int kid)
{
    if (s_shm->kid_res_offset[kid] == NO_RES)
        verror("Multitasking: process %d ended without a result", kid);
    return s_res + s_shm->kid_res_offset[kid];
}

uint64_t PyMisha::get_kid_res_size(int kid)
{
    if (s_shm->kid_res_offset[kid] == NO_RES)
        verror("Multitasking: process %d ended without a result", kid);
    return s_shm->kid_res_num_records[kid];
}

bool     pm::g_rng_loaded = false;
int      pm::g_rng_kid_idx = -1;
bool     pm::g_rng_stream = false;
bool     pm::g_rng_peeked = false;
uint32_t pm::g_rng_peeked_seed = 0;

// A draw from numpy's global generator. The C++ layer holds the GIL throughout a call (it never
// releases it), and draws only in the thread that called it. With restore, numpy's state is put
// back after the draw.
static uint32_t numpy_draw(bool restore)
{
    PMPY random(PyImport_ImportModule("numpy.random"), true);
    PMPY state(random && restore ? PyObject_CallMethod(random, "get_state", NULL) : NULL, true);
    PMPY draw(random && (state || !restore) ?
              PyObject_CallMethod(random, "randint", "K", (unsigned long long)4294967296ULL) : NULL, true);
    PMPY as_long(draw ? PyNumber_Long(draw) : NULL, true);
    unsigned long long seed = as_long ? PyLong_AsUnsignedLongLong(as_long) : 0;
    bool ok = as_long && !PyErr_Occurred();
    if (ok && restore)
        ok = (bool)PMPY(PyObject_CallMethod(random, "set_state", "(O)", (PyObject *)state), true);
    if (!ok) {
        PyErr_Clear();
        verror("Failed to draw a seed from numpy's random generator");
    }
    return (uint32_t)seed;
}

void pm::rng_peek()
{
    g_rng_peeked_seed = numpy_draw(true);
    g_rng_peeked = true;
}

void pm::rng_load()
{
    uint32_t s;
    if (PyMisha::is_kid()) {
        if (!g_rng_peeked)
            verror("A multitasking process needs a random seed its parent did not draw");
        s = g_rng_peeked_seed;
    } else
        s = numpy_draw(false);
    if (g_rng_kid_idx >= 0)
        s = kid_seed(s, g_rng_kid_idx);
    srand48((long)s);
    g_rng_loaded = true;
}

pid_t PyMisha::launch_process()
{
    if (s_shm_sem == SEM_FAILED || s_shm == (Shm *)MAP_FAILED)
        verror("Not ready for multitasking");

    if (s_kid_index >= MAX_KIDS)
        verror("Too many child processes");

    vdebug("SemLock\n");

    check_interrupt();

    {
        SemLocker sl(s_shm_sem);
        if (s_shm->error_msg[0])
            verror("%s", s_shm->error_msg);
    }

    // Within one random stream of several calls, the kids of every call must draw numbers of their
    // own: they seed from the stream's state, which each fork then advances
    if (pm::g_rng_stream && !pm::g_rng_loaded)
        pm::rng_load();

    vdebug("fork\n");

    // Use Python 3.7+ fork handling for proper GIL and threading state management.
    // This is CRITICAL for Jupyter/IPython where ZMQ threads exist.
    // PyOS_BeforeFork() acquires all interpreter locks to ensure safe forking.
    PyOS_BeforeFork();

    pid_t pid = fork();

    if (pid == -1) {
        PyOS_AfterFork_Parent();
        verror("fork failed: %s", strerror(errno));
    }

    if (pid) { // a parent process
        // Release locks acquired by PyOS_BeforeFork()
        PyOS_AfterFork_Parent();
        vdebug("%d: child process %d has been launched\n", getpid(), pid);
        s_running_pids.push_back(pid);
        ++s_kid_index;
        if (pm::g_rng_stream)
            drand48();
    } else {   // a child process
        // A kid dies with its parent. A kid whose parent was killed would otherwise run on to a
        // result nobody reads, or wait forever for the shared memory lock when the kid that held it
        // was killed with the parent.
        // ponytail: Linux only; elsewhere such a kid still hangs - a getppid() poll in its waits would cover it
#ifdef __linux__
        prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
        if (getppid() != s_parent_pid)   // the parent died before prctl()
            _exit(1);

        s_is_kid = true;
        // s_kid_index is this kid's index here (the parent increments it after the fork)
        pm::reseed_kid_rng(s_kid_index);

        // Reinitialize Python's internal state after fork.
        // PyOS_AfterFork_Child() resets threading state, internal locks,
        // and calls registered atfork handlers. This is essential for
        // environments like Jupyter where background threads (ZMQ) exist.
        // It runs the session's os.register_at_fork(after_in_child=...) functions too: Python
        // code runs in every kid as it starts, though its scan does not call into Python.
        PyOS_AfterFork_Child();

        sigaction(SIGINT, &s_old_sigint_act, NULL);
        sigaction(SIGALRM, &s_old_sigalrm_act, NULL);
        sigaction(SIGCHLD, &s_old_sigchld_act, NULL);

        if (!s_multitasking_stdout) {
            if (!freopen("/dev/null", "w", stdout))
                verror("Failed to open /dev/null");
        }

        if (!freopen("/dev/null", "w", stderr))
            verror("Failed to open /dev/null");

        if (!freopen("/dev/null", "r", stdin))
            verror("Failed to open /dev/null");
    }

    return pid;
}

// Drops from s_running_pids the kids that are already gone: one that ended is reaped here, and one
// that someone else reaped (waitpid gives ECHILD) is no longer ours - its pid may even belong to
// another process by now, which must not be signaled. Only these pids are waited for: a
// waitpid(-1) would reap the other children of the Python session too (a subprocess.Popen's), whose
// exit status their owner then never gets.
void PyMisha::check_kids_state(bool ignore_errors)
{
    for (size_t i = 0; i < s_running_pids.size();) {
        int status;
        pid_t pid = s_running_pids[i];
        pid_t res = waitpid(pid, &status, WNOHANG);
        if (res != pid && !(res < 0 && errno == ECHILD)) {
            ++i;
            continue;
        }
        s_running_pids[i] = s_running_pids.back();
        s_running_pids.pop_back();
        if (res == pid && !ignore_errors) {
            if (WIFSIGNALED(status))
                verror("Child process %d ended unexpectedly (signal %d)", (int)pid, WTERMSIG(status));
            if (!WIFEXITED(status))
                verror("Child process %d ended unexpectedly", (int)pid);
        }
    }
}

bool PyMisha::wait_for_kids(int millisecs)
{
    struct timespec timeout, remaining;
    set_rel_timeout(millisecs, timeout);

    while (1) {
        vdebug("SIGINT fired? %d\n", s_sigint_fired);
        check_interrupt();
        check_kids_state(false);

        {
            SemLocker sl(s_shm_sem);
            if (s_shm->error_msg[0])
                verror("%s", s_shm->error_msg);
        }

        // the records of all the kids together, against CONFIG['max_data_size'] (each kid checks
        // its own)
        if (s_res_record_size) {
            uint64_t num_records = 0;
            for (int i = 0; i < s_kid_index; ++i)
                num_records += s_shm->kid_res_num_records[i];
            verify_kids_records(num_records);
        }

        if (s_running_pids.empty()) {
            vdebug("No more running child processes\n");
            return false;
        }

        vdebug("still running %ld child processes (%d, ...)\n", s_running_pids.size(), s_running_pids.front());

        if (nanosleep(&timeout, &remaining))
            timeout = remaining;
        else
            break;
    }
    return true;
}

void PyMisha::wait_for_kids_reporting(PyObject *progress_cb)
{
    const int64_t REPORT_INTERVAL = 3000;   // ms, R misha's REPORT_INTERVAL_DELAY
    struct timespec last_report;
    clock_gettime(CLOCK_REALTIME, &last_report);
    int last_progress = -1;

    while (wait_for_kids(1)) {
        if (!is_time_elapsed(REPORT_INTERVAL, last_report))
            continue;
        // a kid that runs several scans starts each from 0%: what was reported is not taken back
        int progress = 0;
        for (int i = 0; i < s_kid_index; ++i)
            progress += s_shm->kid_progress[i];
        progress = max(progress / max(s_kid_index, 1), last_progress);
        if (progress < 100) {
            if (progress_cb)
                call_progress_cb(progress_cb, progress, 100, progress);
            else if (progress != last_progress)
                vemsg("%d%%...", progress);
            else
                vemsg(".");
        }
        last_progress = progress;
        clock_gettime(CLOCK_REALTIME, &last_report);
    }

    if (progress_cb)
        call_progress_cb(progress_cb, 100, 100, 100);
    else if (last_progress >= 0)
        vemsg("100%%\n");
}

void call_progress_cb(PyObject *progress_cb, uint64_t done, uint64_t total, int pct)
{
    PMPY py_done(PyLong_FromUnsignedLongLong(done), true);
    PMPY py_total(total ? PyLong_FromUnsignedLongLong(total) : Py_NewRef(Py_None), true);
    PMPY py_pct(PyLong_FromLong(pct), true);
    if (!py_done || !py_total || !py_pct)
        TGLError("Failed to allocate arguments for progress callback");

    PMPY res(PyObject_CallFunctionObjArgs(progress_cb, (PyObject *)py_done, (PyObject *)py_total, (PyObject *)py_pct,
                                          NULL), true);
    if (!res || PyErr_Occurred())
        TGLError("Progress callback failed");
}

void PyMisha::handle_error(const char *msg)
{
    if (s_is_kid) {
        {
            SemLocker sl(s_shm_sem);
            if (!s_shm->error_msg[0]) {
                snprintf(s_shm->error_msg, sizeof(s_shm->error_msg), "%s", msg);
            }
        }
        // Use _exit() instead of exit() to avoid running Python/C++ destructors
        // which can corrupt ZMQ sockets in Jupyter notebooks
        _exit(1);
    } else
        PyErr_SetString(s_pm_err, msg);
}

void PyMisha::verify_max_data_size(uint64_t data_size, const char *data_name, const char *unit)
{
    // CONFIG['max_data_size'] is a number of records (rows of a result, values, bases), not of
    // bytes. data_size is often the number reached so far, not the final one: no value to suggest.
    if (data_size > max_data_size())
        verror("%s size (at least %llu %s) exceeded the maximum allowed (%llu %s, set by CONFIG['max_data_size']).\n\n"
               "Suggestions:\n"
               "  - Raise the limit: CONFIG['max_data_size'] = <a larger number of %s>, or None for no limit\n"
               "  - Reduce scope (use smaller genomic regions or filters)\n"
               "  - Process data in smaller chunks",
               data_name && *data_name ? data_name : "Result", (unsigned long long)data_size, unit,
               (unsigned long long)max_data_size(), unit, unit);
}

void PyMisha::set_alarm(int msecs)
{
    struct itimerval timer;

    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;

    timer.it_value.tv_sec = msecs / 1000;
    timer.it_value.tv_usec = (msecs % 1000) * 1000;

    setitimer(ITIMER_REAL, &timer, NULL);
}

void PyMisha::reset_alarm()
{
    s_sigalrm_fired = 0;

    struct itimerval timer;

    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 0;

    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 0;

    setitimer(ITIMER_REAL, &timer, NULL);
}

void PyMisha::load_options()
{
    PMPY module, dict, opts, opt;

    if (!module.assign(PyImport_AddModule("pymisha"), false) ||
        !dict.assign(PyModule_GetDict(module), false) ||
        !opts.assign(PyDict_GetItem(dict, PMPY(PyUnicode_FromString("CONFIG"), true)), false))
        return;

    // An invalid value is this call's error; no Python exception is left set
    auto python_error = []() {
        std::string msg = "Invalid CONFIG value";
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        if (value) {
            PMPY str(PyObject_Str(value), true);
            if (str && PyUnicode_AsUTF8(str))
                msg = PyUnicode_AsUTF8(str);
        }
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        PyErr_Clear();
        verror("%s", msg.c_str());
    };
    // A number of any type (int, float, a numpy scalar) truncated toward zero, as R misha reads
    // gmax.processes and the like, and as int() does: 8.0, 8.7 and numpy.int64(8) are all 8. False
    // for a value of another type, which leaves the default.
    auto as_long = [&](const char *name, long &v) {
        if (!PyNumber_Check(opt))
            return false;
        PMPY num(PyNumber_Long(opt), true);
        if (!num) {   // NaN, an infinity
            PyErr_Clear();
            verror("CONFIG['%s'] must be a finite number", name);
        }
        v = PyLong_AsLong(num);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            verror("CONFIG['%s'] is out of range", name);
        }
        return true;
    };
    long v;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("debug"), true)), false) && PyBool_Check(opt))
        m_debug = opt == Py_True;

    // multitasking, max_data_size and max_mem_usage are read by the functions the Python side
    // reads them with, which reject an invalid value as R misha does
    PMPY name(PyUnicode_FromString("pymisha._shared"), true);
    PMPY shared(PyImport_GetModule(name), true);   // imported with pymisha, whose CONFIG is read
    if (!shared)
        python_error();
    PMPY multitasking(PyObject_CallMethod(shared, "_config_multitasking", NULL), true);
    if (!multitasking)
        python_error();
    m_multitasking_avail = multitasking == Py_True;
    PMPY max_data_size(PyObject_CallMethod(shared, "_config_limit", "s", "max_data_size"), true);
    if (!max_data_size)
        python_error();
    m_max_data_size = PyLong_AsUnsignedLongLong(max_data_size);
    if (PyErr_Occurred())
        python_error();
    PMPY max_mem_usage(PyObject_CallMethod(shared, "_config_limit", "s", "max_mem_usage"), true);
    if (!max_mem_usage)
        python_error();
    // in KB; R misha counts 1000 bytes to the KB
    uint64_t max_mem_usage_kb = PyLong_AsUnsignedLongLong(max_mem_usage);
    if (PyErr_Occurred())
        python_error();
    m_max_mem_usage = max_mem_usage_kb >= MAX_DATA_SIZE_NO_LIMIT ? ~(uint64_t)0 : max_mem_usage_kb * 1000;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("multitasking_stdout"), true)), false) && PyBool_Check(opt))
        s_multitasking_stdout = opt == Py_True;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("max_processes"), true)), false) &&
        as_long("max_processes", v))
        m_max_processes = max((int)v, 1);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("min_scope4process"), true)), false) &&
        as_long("min_scope4process", v))
        m_min_scope4process = (uint64_t)max(v, 0L);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("eval_buf_size"), true)), false) &&
        as_long("eval_buf_size", v))
        m_eval_buf_size = max(v, 1L);
}

void PyMisha::sigint_handler(int)
{
    ++s_sigint_fired;

    if (getpid() == s_parent_pid)
        vemsg("CTL-C!\n");
}

void PyMisha::sigalrm_handler(int)
{
    s_sigalrm_fired = true;
}

void PyMisha::sigchld_handler(int)
{
}

void PyMisha::set_dicts4regression_tests(PyObject *gdict, PyObject *c_dict, PyObject *py_dict)
{
    m_gdict.assign(gdict);
    m_c_module_dict.assign(c_dict);
    m_py_module_dict.assign(py_dict);
}

void vmsg(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);

    fflush(stdout);
}

void vemsg(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fflush(stderr);
}

[[noreturn]] void verror(const char *fmt, ...)
{
    va_list ap;
    char buf[1000];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    // A kid reports through shared memory and _exit(1)s, so handle_error() never
    // returns there.
    if (PyMisha::s_is_kid)
        PyMisha::handle_error(buf);

    // The parent always throws. This used to branch on s_ref_count and call
    // handle_error() when it was 0 - which sets the Python error and RETURNS, so
    // the caller ran on with an exception set and invalid state. Every registered
    // entry point catches TGLException and converts it, so throwing is what the
    // 432 call sites already assume; the s_ref_count == 0 branch was the odd one
    // out. See handle_error(), still used from those catch blocks.
    TGLError("%s", buf);
}

void vwarning(const char *fmt, ...)
{
    va_list ap;
    char buf[1000];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    PyErr_WarnExplicit(PyExc_Warning, buf, "pymisha", 0, NULL, NULL);
}

void vdebug(const char *fmt, ...)
{
    if (g_pymisha && g_pymisha->debug()) {
        struct timeval tmnow;
        struct tm *tm;
        char buf[100];
        gettimeofday(&tmnow, NULL);
        tm = localtime(&tmnow.tv_sec);
        strftime(buf, sizeof(buf), "%H:%M:%S", tm);
        vemsg("[DEBUG %s.%03d] ", buf, (int)(tmnow.tv_usec / 1000));

        va_list ap;
        va_start(ap, fmt);
        vprintf(fmt, ap);
        va_end(ap);

        if (!*fmt || (*fmt && fmt[strlen(fmt) - 1] != '\n'))
            vemsg("\n");

        fflush(stderr);
    }
}

