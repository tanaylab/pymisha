#ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE 199309
    #include <time.h>
    #undef _POSIX_C_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
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
    if (!s_ref_count) {
        m_old_umask = umask(07);

        ++g_transact_id;

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

        g_pymisha = this;
        m_old_error_handler = TGLException::set_error_handler(TGLException::throw_error_handler);

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

    }

    s_ref_count++;

    if (check_db) {
        if (s_groot.empty())
            verror("Database was not loaded. Please call gdb.init.");
    }
}

PyMisha::~PyMisha()
{
    s_ref_count--;

    if (!s_ref_count) {
        // if this is a child, do not detach from shared memory and do not deallocate the semaphore
        if (!s_is_kid) {
            if (s_shm_sem != SEM_FAILED) {
                SemLocker sl(s_shm_sem);
                SigBlocker sb;

                // kill all the remaining child processes
                for (vector<pid_t>::const_iterator ipid = s_running_pids.begin(); ipid != s_running_pids.end(); ++ipid) {
                    vdebug("Forcefully terminating process %d\n", *ipid);
                    kill(*ipid, SIGTERM);
                }
            }

            // after SIGTERM is sent to all the kids let's wait till sigchld_hander() burries them all
            struct timespec timeout;
            while (1) {
                check_kids_state(true);
                if (s_running_pids.empty())
                    break;

                vdebug("Waiting for %ld child processes to end\n", s_running_pids.size());
                
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
    sprintf(buf, "pymisha-shm-sem-%d", (int)getpid());
    return buf;
}

string PyMisha::get_fifo_sem_name()
{
    char buf[100];
    sprintf(buf, "pymisha-fifo-sem-%d", (int)getpid());
    return buf;
}

string PyMisha::get_fifo_name()
{
    char buf[100];
    sprintf(buf, "/tmp/pymisha-fifo-%d", s_is_kid ? (int)getppid() : (int)getpid());
    return buf;
}

void PyMisha::prepare4multitasking()
{
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
    } else {   // a child process
        s_is_kid = true;

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

        if (eof_reached && s_running_pids.empty())
            return readlen;
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
                strncpy(s_shm->error_msg, msg, sizeof(s_shm->error_msg) - 1);
                s_shm->error_msg[sizeof(s_shm->error_msg) - 1] = '\0';
            }
        }
        // Use _exit() instead of exit() to avoid running Python/C++ destructors
        // which can corrupt ZMQ sockets in Jupyter notebooks
        _exit(1);
    } else
        PyErr_SetString(s_pm_err, msg);
}

void PyMisha::verify_max_data_size(uint64_t data_size, const char *data_name)
{
    if (data_size > max_data_size())
        verror("%s size exceeded the maximal allowed (%ld).\n"
               "Note: the maximum data size is controlled via CONFIG['max_data_size'] option.",
               data_name, max_data_size());
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

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("debug"), true)), false) && PyBool_Check(opt))
        m_debug = opt == Py_True;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("multitasking"), true)), false) && PyBool_Check(opt))
        m_multitasking_avail = opt == Py_True;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("multitasking_stdout"), true)), false) && PyBool_Check(opt))
        s_multitasking_stdout = opt == Py_True;

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("min_processes"), true)), false) && PyLong_Check(opt))
        m_min_processes = max(PyLong_AsLong(opt), 1L);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("max_processes"), true)), false) && PyLong_Check(opt))
        m_max_processes = max((int)PyLong_AsLong(opt), m_min_processes);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("max_data_size"), true)), false) && PyLong_Check(opt))
        m_max_data_size = max(PyLong_AsUnsignedLong(opt), 1UL);

    if (*opt.assign(PyDict_GetItem(opts, PMPY(PyUnicode_FromString("eval_buf_size"), true)), false) && PyLong_Check(opt))
        m_eval_buf_size = max(PyLong_AsUnsignedLong(opt), 1UL);
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

void verror(const char *fmt, ...)
{
    va_list ap;
    char buf[1000];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (PyMisha::s_ref_count)
        TGLError("%s", buf);
    else
        PyMisha::handle_error(buf);
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

string get_bound_colname(const char *str, unsigned maxlen)
{
    string colname;

    maxlen = max(maxlen, 4u);
    if (strlen(str) > maxlen) {
        colname.assign(str, maxlen - 3);
        colname += "...";
    } else
        colname = str;
    return colname;
}
