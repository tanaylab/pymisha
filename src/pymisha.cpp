#include <time.h>

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <algorithm>
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
sem_t                   *PyMisha::s_fifo_sem = SEM_FAILED;
int                      PyMisha::s_kid_index;
vector<pid_t>            PyMisha::s_running_pids;
PyMisha::Shm            *PyMisha::s_shm = (PyMisha::Shm *)MAP_FAILED;
int                      PyMisha::s_fifo_fd = -1;
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

        s_sigint_fired = 0;
        s_sigalrm_fired = 0;
        s_is_kid = false;
        s_kid_index = 0;
        s_parent_pid = getpid();
        s_shm_sem = SEM_FAILED;
        s_fifo_sem = SEM_FAILED;
        s_shm = (Shm *)MAP_FAILED;
        s_fifo_fd = -1;
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
            forget_reaped_kids();

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
            // waitpid() reaped is no longer ours and is dropped (waitpid(-1) never returns it, so
            // the wait would not end). A kid still alive after 10 s gets SIGKILL; after 30 s the
            // rest are given up on (as R misha 5.17).
            struct timespec timeout;
            for (int waited_ms = 0;; waited_ms += 10) {
                check_kids_state(true);
                forget_reaped_kids();
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

            if (s_fifo_sem != SEM_FAILED)
                sem_close(s_fifo_sem);

            if (s_shm != (Shm *)MAP_FAILED)
                munmap(s_shm, sizeof(Shm));

            unlink(get_fifo_name().c_str());
        }

        if (s_fifo_fd != -1)
            close(s_fifo_fd);

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

string PyMisha::get_fifo_sem_name()
{
    char buf[100];
    snprintf(buf, sizeof(buf), "pymisha-fifo-sem-%d", (int)getpid());
    return buf;
}

string PyMisha::get_fifo_name()
{
    char buf[100];
    snprintf(buf, sizeof(buf), "/tmp/pymisha-fifo-%d", s_is_kid ? (int)getppid() : (int)getpid());
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

    vdebug("Cleaning old semaphores\n");
    if (s_shm_sem == SEM_FAILED) {
        sem_unlink(get_shm_sem_name().c_str());
        if ((s_shm_sem = sem_open(get_shm_sem_name().c_str(), O_CREAT | O_EXCL, 0644, 1)) == SEM_FAILED)
            verror("sem_open failed: %s", strerror(errno));

        sem_unlink(get_shm_sem_name().c_str());
    }

    if (s_fifo_sem == SEM_FAILED) {
        sem_unlink(get_fifo_sem_name().c_str());
        if ((s_fifo_sem = sem_open(get_fifo_sem_name().c_str(), O_CREAT | O_EXCL, 0644, 1)) == SEM_FAILED)
            verror("sem_open failed: %s", strerror(errno));

        sem_unlink(get_fifo_sem_name().c_str());
    }

    vdebug("Creating FIFO channel\n");
    if (s_fifo_fd == -1) {
        unlink(get_fifo_name().c_str());

        if (mkfifo(get_fifo_name().c_str(), 0666) == -1)
            verror("mkfifo of file %s failed: %s", get_fifo_name().c_str(), strerror(errno));

        if ((s_fifo_fd = open(get_fifo_name().c_str(), O_RDONLY | O_NONBLOCK)) == -1)
            verror("open of fifo %s for read failed: %s", get_fifo_name().c_str(), strerror(errno));

#ifdef F_SETPIPE_SZ
        fcntl(s_fifo_fd, F_SETPIPE_SZ, 1048576);
#endif
    }

    vdebug("Allocating shared memory for internal communication\n");
    if (s_shm == (Shm *)MAP_FAILED) {
        s_shm = (Shm *)mmap(NULL, sizeof(Shm), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

        if (s_shm == (Shm *)MAP_FAILED)
            verror("Failed to allocate shared memory: %s", strerror(errno));

        s_shm->error_msg[0] = '\0';
        for (int i = 0; i < MAX_KIDS; ++i)
            s_shm->itr_idx[i] = 0;
    }
}

bool pm::g_rng_loaded = false;
int  pm::g_rng_kid_idx = -1;
bool pm::g_rng_stream = false;

void pm::rng_load()
{
    // A draw from numpy's global generator. The C++ layer holds the GIL throughout a call (it never
    // releases it), and draws only in the thread that called it.
    PMPY random(PyImport_ImportModule("numpy.random"), true);
    PMPY draw(random ? PyObject_CallMethod(random, "randint", "K", (unsigned long long)4294967296ULL) : NULL, true);
    PMPY as_long(draw ? PyNumber_Long(draw) : NULL, true);
    unsigned long long seed = as_long ? PyLong_AsUnsignedLongLong(as_long) : 0;
    if (!as_long || PyErr_Occurred()) {
        PyErr_Clear();
        verror("Failed to draw a seed from numpy's random generator");
    }
    uint32_t s = (uint32_t)seed;
    if (g_rng_kid_idx >= 0)
        s = kid_seed(s, g_rng_kid_idx);
    srand48((long)s);
    g_rng_loaded = true;
}

pid_t PyMisha::launch_process()
{
    if (s_shm_sem == SEM_FAILED || s_fifo_sem == SEM_FAILED || s_shm == (Shm *)MAP_FAILED || s_fifo_fd == -1)
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
        s_is_kid = true;
        // s_kid_index is this kid's index here (the parent increments it after the fork)
        pm::reseed_kid_rng(s_kid_index);

        // Reinitialize Python's internal state after fork.
        // PyOS_AfterFork_Child() resets threading state, internal locks,
        // and calls registered atfork handlers. This is essential for
        // environments like Jupyter where background threads (ZMQ) exist.
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

        close(s_fifo_fd);

        if ((s_fifo_fd = open(get_fifo_name().c_str(), O_WRONLY)) == -1)
            verror("open of fifo %s for write failed: %s", get_fifo_name().c_str(), strerror(errno));
    }

    return pid;
}

void PyMisha::check_kids_state(bool ignore_errors)
{
    int status;
    pid_t pid;

    while ((pid = waitpid((pid_t)-1, &status, WNOHANG)) > 0) {
        vdebug("pid %d has ended\n", pid);
        for (vector<pid_t>::iterator ipid = s_running_pids.begin(); ipid != s_running_pids.end(); ++ipid) {
            if (*ipid == pid) {
                vdebug("pid %d was identified as a child process\n", pid);
                swap(*ipid, s_running_pids.back());
                s_running_pids.pop_back();
                if (!ignore_errors && !WIFEXITED(status))
                    verror("Child process %d ended unexpectedly", (int)pid);
                break;
            }
        }
    }
}

// Drops from s_running_pids the kids that are already gone: one that ended is reaped here, and one
// that someone else reaped (waitpid gives ECHILD) is no longer ours - its pid may even belong to
// another process by now, which must not be signaled.
void PyMisha::forget_reaped_kids()
{
    for (size_t i = 0; i < s_running_pids.size();) {
        int status;
        pid_t res = waitpid(s_running_pids[i], &status, WNOHANG);
        if (res == s_running_pids[i] || (res < 0 && errno == ECHILD)) {
            s_running_pids[i] = s_running_pids.back();
            s_running_pids.pop_back();
        } else
            ++i;
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

int PyMisha::read_multitask_fifo(void *buf, size_t bytes)
{
    bool eof_reached = false;
    int retv;
    size_t readlen = 0;
    fd_set rfds;
    struct timeval tv;

    while (bytes > readlen) {
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        FD_ZERO(&rfds);
        FD_SET(s_fifo_fd, &rfds);

        retv = select(s_fifo_fd + 1, &rfds, NULL, NULL, &tv);

        if (retv == -1) {
            if (errno != EINTR)
                verror("select on fifo failed: %s", strerror(errno));
        } else if (retv == 1) {
            retv = read(s_fifo_fd, buf, bytes - readlen);

            if (retv == -1) {
                if (errno != EAGAIN && errno != EWOULDBLOCK)
                    verror("read from fifo failed: %s", strerror(errno));
            } else {
                buf = (char *)buf + retv;
                readlen += retv;

                if (!retv)
                    eof_reached = true;
            }
        }

        check_interrupt();

        if (s_shm->error_msg[0]) {
            SemLocker sl(s_shm_sem);
            verror("%s", s_shm->error_msg);
        }

        check_kids_state(false);

        // Every caller reads a fixed-size message and none checks the length,
        // so running out of data is an error, not a short read.
        if (eof_reached && s_running_pids.empty() && readlen < bytes)
            verror("Multitasking: a child process ended before sending all of its result "
                   "(%lu of %lu bytes)", (unsigned long)readlen, (unsigned long)bytes);
    }
    return readlen;
}

void PyMisha::write_multitask_fifo(const void *buf, size_t bytes)
{
    SemLocker sl(s_fifo_sem);
    const char *ptr = (const char *)buf;
    size_t remaining = bytes;
    while (remaining > 0) {
        ssize_t written = write(s_fifo_fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            verror("write to fifo failed: %s", strerror(errno));
        }
        if (written == 0) {
            verror("write to fifo failed: wrote 0 bytes");
        }
        ptr += written;
        remaining -= (size_t)written;
    }
}

void PyMisha::lock_multitask_fifo()
{
    if (s_fifo_sem == SEM_FAILED)
        verror("Not ready for multitasking");
    if (sem_wait(s_fifo_sem) == -1)
        verror("sem_wait failed: %s", strerror(errno));
}

void PyMisha::unlock_multitask_fifo()
{
    if (s_fifo_sem == SEM_FAILED)
        verror("Not ready for multitasking");
    if (sem_post(s_fifo_sem) == -1)
        verror("sem_post failed: %s", strerror(errno));
}

void PyMisha::write_multitask_fifo_unlocked(const void *buf, size_t bytes)
{
    const char *ptr = (const char *)buf;
    size_t remaining = bytes;
    while (remaining > 0) {
        ssize_t written = write(s_fifo_fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            verror("write to fifo failed: %s", strerror(errno));
        }
        if (written == 0) {
            verror("write to fifo failed: wrote 0 bytes");
        }
        ptr += written;
        remaining -= (size_t)written;
    }
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
    auto as_long = [&](const char *name) {
        long v = PyLong_AsLong(opt);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            verror("CONFIG['%s'] is out of range", name);
        }
        return v;
    };

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
    if (!PMPY(PyObject_CallMethod(shared, "_config_limit", "s", "max_mem_usage"), true))
        python_error();

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("multitasking_stdout"), true)), false) && PyBool_Check(opt))
        s_multitasking_stdout = opt == Py_True;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("min_processes"), true)), false) && PyLong_Check(opt))
        m_min_processes = max(as_long("min_processes"), 1L);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("max_processes"), true)), false) && PyLong_Check(opt))
        m_max_processes = max((int)as_long("max_processes"), m_min_processes);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("min_scope4process"), true)), false) && PyLong_Check(opt))
        m_min_scope4process = (uint64_t)max(as_long("min_scope4process"), 0L);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("min_intervs4process"), true)), false) && PyLong_Check(opt))
        m_min_intervs4process = (uint64_t)max(as_long("min_intervs4process"), 0L);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("eval_buf_size"), true)), false) && PyLong_Check(opt))
        m_eval_buf_size = max(as_long("eval_buf_size"), 1L);
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

