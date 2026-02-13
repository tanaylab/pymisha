#ifndef PYMISHA_H_
#define PYMISHA_H_

// This must be defined before in all the files that use numpy before "#include <numpy/arrayobject.h>"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pymisha_ARRAY_API

#include <string>
#include <set>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "PMObject.h"
#include "TGLException.h"

#ifndef PYPYMISHA
    #define PYPYMISHA
#endif

using namespace std;

// Throws exception if the command is interrupted.
inline void check_interrupt();

// adds timeout to the time that is already in req
void set_abs_timeout(int64_t delay_msec, struct timespec &req);

// sets timeout to req
void set_rel_timeout(int64_t delay_msec, struct timespec &req);

// returns true if current time exceeds start_time + delay
bool is_time_elapsed(int64_t delay_msec, const struct timespec &start_time);

void vmsg(const char *fmt, ...);
void vemsg(const char *fmt, ...);

void verror(const char *fmt, ...);

void vwarning(const char *fmt, ...);

void vdebug(const char *fmt, ...);

#define DBGHERE vemsg("%s, line %d\n", __FILE__, __LINE__);

inline bool is_py_var_char(char c) { return isalnum(c) || c == '_' || c == '.'; }

string get_bound_colname(const char *str, unsigned maxlen = 40);

template<typename T> void pack_data(void *&ptr, const T &data, size_t n) {
        size_t size = sizeof(data) * n;
        memcpy(ptr, &data, size);
        ptr = (char *)ptr + size;
}

template<typename T> void unpack_data(void *&ptr, T &data, size_t n) {
        size_t size = sizeof(data) * n;
        memcpy(&data, ptr, size);
        ptr = (char *)ptr + size;
}


#define MAX_KIDS 1000

// returning PMPY object
// Note: Use _exit() instead of exit() in child processes to avoid running
// Python/C++ destructors which can corrupt ZMQ sockets in Jupyter notebooks
#define return_py(retv) {     \
    if (PyMisha::is_kid())    \
        _exit(0);             \
    PMPY obj = retv;          \
    obj.to_be_stolen();       \
    return (PyObject *)obj;   \
}

// returning no value
#define return_none() {       \
    if (PyMisha::is_kid())    \
        _exit(0);             \
    Py_INCREF(Py_None);       \
    return Py_None;           \
}

// returning whilst error
#define return_err() {        \
    if (PyMisha::is_kid())    \
        _exit(0);             \
    return NULL;              \
}

inline double     PMDOUBLE(const PMPY &obj, size_t i) { return *(double *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline long       PMLONG(const PMPY &obj, size_t i) { return *(long *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline bool       PMBOOL(const PMPY &obj, size_t i) { return *(bool *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline PyObject  *PMOBJ(const PMPY &obj, size_t i) { return *(PyObject **)PyArray_GETPTR1((PyArrayObject *)*obj, i); }

inline double    &PMDOUBLE(PMPY &obj, size_t i) { return *(double *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline long      &PMLONG(PMPY &obj, size_t i) { return *(long *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline bool      &PMBOOL(PMPY &obj, size_t i) { return *(bool *)PyArray_GETPTR1((PyArrayObject *)*obj, i); }
inline PyObject *&PMOBJ(PMPY &obj, size_t i) { return *(PyObject **)PyArray_GETPTR1((PyArrayObject *)*obj, i); }

inline bool       PMIS1D(const PMPY &obj) { return *obj && PyArray_Check((PyArrayObject *)*obj) && PyArray_NDIM((PyArrayObject *)*obj) == 1; }
inline size_t     PMLEN(const PMPY &obj) { return PyArray_DIM((PyArrayObject *)*obj, 0); }

inline bool       PMISSTRLIST(const PMPY &obj) {
    if (!obj || !PyList_Check(obj) || !PyList_Size(obj))
        return false;
    for (int i = 0; i < PyList_Size(obj); ++i) {
        if (!PyUnicode_Check(PyList_GetItem(obj, i)))
            return false;
    }
    return true;
}

// Define PyMisha instance in your main function that is called by Python.
// PyMisha should be defined inside "try-catch" statement that catches TGLException.
// PyMisha performs the following actions:
//   1. Installs a new SIGINT handler. ONE MUST CALL check_interrupt()
//   2. Installs out-of-memory handler.
//   3. Supresses the default error report behaviour.
//   4. Makes sure all file descriptors are closed on exit / error / interrupt.
//   5. Makes sure all objects are destructed on exit / error / interrupt.

class PyMisha {
public:
    PyMisha(bool check_db = true);
    ~PyMisha();

    const PMPY &gdict() const { return m_gdict; }
    const PMPY &c_module_dict() const { return m_c_module_dict; }
    const PMPY &py_module_dict() const { return m_py_module_dict; }

    // Verifies that the data size does not exceed the maximum allowed.
    void verify_max_data_size(uint64_t data_size, const char *data_name = "Result");

    // true if debug prints are allowed
    bool debug() const { return m_debug; }

    // Returns true if multitasking option is switched on
    bool multitasking_avail() const { return m_multitasking_avail; }

    // Returns min / max number of processes
    int min_processes() const { return m_min_processes; }
    int max_processes() const { return m_max_processes; }

    // Returns the upper limit for data size
    uint64_t max_data_size() const { return m_max_data_size; }

    // Returns buffer size of the numpy arrays used within PyEval_EvalCode
    int eval_buf_size() const { return m_eval_buf_size; }

    void set_dicts4regression_tests(PyObject *gdict, PyObject *c_dict, PyObject *py_dict);

    static void handle_error(const char *msg);

    static void set_alarm(int msecs);   // time is given in milliseconds
    static void reset_alarm();
    static int alarm_fired() { return s_sigalrm_fired; }

    static void prepare4multitasking();
    static pid_t launch_process();

    // returns false if all the child processes have ended or true if the timeout has elapsed
    static bool wait_for_kids(int millisecs);

    // returns number of bytes read or 0 for EOF; the parent process that uses fifo does not need to call then wait_for_kids()
    static int read_multitask_fifo(void *buf, size_t bytes);
    static void write_multitask_fifo(const void *buf, size_t bytes);
    static void lock_multitask_fifo();
    static void unlock_multitask_fifo();
    static void write_multitask_fifo_unlocked(const void *buf, size_t bytes);

    static bool is_kid() { return s_is_kid; }

    static void itr_idx(uint64_t idx) { s_shm->itr_idx[s_kid_index] = idx; }

    static uint64_t itr_idx_sum();   // sum of itr_idx over all the kids

    static int num_kids() { return s_kid_index; }

    // Database path accessors (static so they persist across PyMisha instances)
    static const string &groot() { return s_groot; }
    static const string &uroot() { return s_uroot; }
    static void set_db_paths(const string &groot, const string &uroot) {
        s_groot = groot;
        s_uroot = uroot;
    }

protected:
    struct Shm {
        char          error_msg[10000];
        uint64_t      itr_idx[MAX_KIDS];          // used for progress report
    };

    struct SigBlocker {
        SigBlocker() {
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGCHLD);
            sigaddset(&sigset, SIGINT);
            sigprocmask(SIG_BLOCK, &sigset, &oldsigset);
        }

        ~SigBlocker() { sigprocmask(SIG_UNBLOCK, &sigset, NULL); }

        sigset_t sigset;
        sigset_t oldsigset;
    };

    static struct sigaction     s_old_sigint_act;
    static struct sigaction     s_old_sigalrm_act;
    static struct sigaction     s_old_sigchld_act;
    static int                  s_ref_count;
    static int                  s_sigint_fired;
    static bool                 s_sigalrm_fired;

    static bool                 s_is_kid;
    static pid_t                s_parent_pid;
    static sem_t               *s_shm_sem;
    static sem_t               *s_fifo_sem;
    static int                  s_kid_index;
    static vector<pid_t>        s_running_pids;
    static Shm                 *s_shm;
    static int                  s_fifo_fd;

    PMPY                        m_gdict;
    PMPY                        m_c_module_dict;
    PMPY                        m_py_module_dict;
    mode_t                      m_old_umask;
    TGLException::Error_handler m_old_error_handler;

    bool                        m_debug{false};
    bool                        m_multitasking_avail{false};
    int                         m_min_processes{4};
    int                         m_max_processes{20};
    uint64_t                    m_max_data_size{10000000};
    int                         m_eval_buf_size{1000};
    static bool                 s_multitasking_stdout;

    // Database paths (static so they persist across PyMisha instances)
    static string               s_groot;
    static string               s_uroot;

    void load_options();

    static string  get_shm_sem_name();
    static string  get_fifo_sem_name();
    static string  get_fifo_name();
    static void    sigint_handler(int);
    static void    sigalrm_handler(int);
    static void    sigchld_handler(int);
    static void    check_kids_state(bool ignore_errors);

    friend void check_interrupt();
    friend void verror(const char *fmt, ...);
};

extern PyMisha    *g_pymisha;
extern PyObject *g_module;


// ------------------------------- IMPLEMENTATION --------------------------------

inline void check_interrupt()
{
    if (PyErr_CheckSignals() == -1)
        TGLError("Command interrupted!");
}

inline void set_abs_timeout(int64_t delay_msec, struct timespec &req)
{
        req.tv_nsec += delay_msec * 1000000L;
        req.tv_sec += req.tv_nsec / 1000000000L;
        req.tv_nsec %= 1000000000L;
}

inline void set_rel_timeout(int64_t delay_msec, struct timespec &req)
{
        req.tv_sec = delay_msec / 1000;
        req.tv_nsec = (delay_msec - req.tv_sec * 1000) * 1000000L;
}

inline bool is_time_elapsed(int64_t delay_msec, const struct timespec &start_time)
{
        struct timespec t1 = start_time;
        struct timespec t2;
        set_abs_timeout(delay_msec, t1);
        clock_gettime(CLOCK_REALTIME, &t2);
        return t2.tv_sec > t1.tv_sec || (t2.tv_sec == t1.tv_sec && t2.tv_nsec > t1.tv_nsec);
}

inline uint64_t PyMisha::itr_idx_sum()
{
    uint64_t res = 0;
    for (int i = 0; i < s_kid_index; ++i)
        res += s_shm->itr_idx[i];
    return res;
}

#endif
