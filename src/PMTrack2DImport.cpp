// gtrack_2d_import and gtrack_2d_import_contacts.
//
// Port of R misha's GTrack2DImport.cpp and GTrack2DImportContacts.cpp (5.12.1, the parallel
// import). Both run in three stages:
//   1. read the input files and split the records into intermediate binary files, one per
//      chromosome pair and kid;
//   2. split each chromosome pair whose quad tree has several subtrees (see pair_num_subtrees)
//      into one file per subtree, so that its memory can be estimated by its largest subtree;
//   3. build the quad tree of each chromosome pair from its intermediate or subtree files.
// With multitasking on, stage 1 runs in kids that each read a contiguous block of the input files,
// and stages 2 and 3 in kids that each take the next chromosome pair, one pair at a time. A pair's
// intermediate files are read back in kid order, which is input file order, so every quad tree gets
// its records in the same order as a serial run and the track files come out byte-identical,
// whichever kid splits or builds the pair.
//
// pymisha differences: the track directory is made by the caller; a pair file is written as
// PMTrack2DCreate.cpp writes it (signature, then the serializer); R's progress report and the
// memory throttle of the stage 1 and 2 kids are not ported. The entry points return the chromosome
// names of the input that the database does not have and leave the checks and the report of them
// (R's verify_chroms_matched and report_skipped_chroms) to the caller, which also says why nothing was
// imported when no chromosome pair got data.

#include "pymisha.h"

#include <dirent.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BufferedFile.h"
#include "GenomeTrack.h"
#include "HashFunc.h"
#include "PMDb.h"
#include "PMTrackExpression2DIterator.h"
#include "StatQuadTreeCached.h"
#include "StatQuadTreeCachedSerializer.h"
#include "TGLException.h"
#include "UnknownChroms.h"

using namespace std;

namespace {

// R misha's gtrack.chunk.size and gtrack.num.chunks defaults (ConfigurationDefaults.h)
const int64_t TRACK_CHUNK_SIZE = 100000;
const int64_t TRACK_NUM_CHUNKS = 0;

// GenomeTrack::FORMAT_SIGNATURES of RECTS and POINTS
const int32_t RECTS_SIGNATURE = -9;
const int32_t POINTS_SIGNATURE = -10;

// R misha's misha::config::MIN_RECORDS_PER_PROCESS and the gmax.processes2core that misha sets at load
const uint64_t MIN_RECORDS_PER_PROCESS = 1000;
const uint64_t MAX_PROCESSES2CORE = 2;

// GInterval2D::COL_NAMES
enum { CHROM1, START1, END1, CHROM2, START2, END2, NUM_2D_COLS };
const char *COL_NAMES_2D[NUM_2D_COLS] = { "chrom1", "start1", "end1", "chrom2", "start2", "end2" };

typedef pair<uint32_t, uint32_t> Chrom_pair;

struct PairFiles {
    int            chromid1;
    int            chromid2;
    int64_t        size;   // total size of the intermediate files in bytes
    vector<string> files;  // intermediate files ordered by kid
};

// The arguments of an import that every stage reads
struct ImportArgs {
    string                   dirname;
    vector<string>           files;
    uint64_t                 max_data_size;
    uint64_t                 max_mem_usage;  // bytes
    const GenomeChromKey    *chromkey;
};

// strutil.cpp split_line(BufferedFile &, ...): returns the number of lines read; fields is empty at EOF
int split_line(BufferedFile &in, vector<string> &fields, char delim)
{
    int num_lines = 0;
    fields.resize(1);
    fields[0].resize(0);
    vector<string>::iterator istr = fields.begin();
    while (1) {
        int c = in.getc();

        if (in.error()) {
            fields.clear();
            break;
        }

        if (c == '\r')
            continue;

        if (c == '\n' || in.eof()) {
            num_lines++;
            if (istr == fields.begin() && istr->empty()) {
                if (in.eof()) {
                    fields.clear();
                    break;
                }

                // eat up empty lines
                continue;
            }
            fields.resize(istr - fields.begin() + 1);
            break;
        }

        if (c == delim) {
            istr++;
            if (istr == fields.end()) {
                fields.push_back(string());
                istr = fields.begin() + fields.size() - 1;
            }
        } else
            istr->push_back(c);
    }
    return num_lines;
}

void open_chk(BufferedFile &file, const string &path, const char *mode)
{
    if (file.open(path.c_str(), mode))
        TGLError("Opening file %s: %s", path.c_str(), strerror(errno));
}

void write_signature(BufferedFile &file, const string &path, int32_t signature)
{
    open_chk(file, path, "wb");
    if (file.write(&signature, sizeof(signature)) != sizeof(signature))
        TGLError("Failed to write to file %s: %s", path.c_str(), strerror(errno));
}

void close_chk(BufferedFile &file)
{
    if (file.close())
        TGLError("Failed to write to file %s: %s", file.file_name().c_str(), strerror(errno));
}

vector<int64_t> get_files_sizes(const vector<string> &files)
{
    vector<int64_t> sizes;

    for (const string &fname : files) {
        struct stat st;

        if (stat(fname.c_str(), &st))
            verror("Accessing file %s: %s", fname.c_str(), strerror(errno));
        sizes.push_back(st.st_size);
    }
    return sizes;
}

//------------------------------- SHARED BY BOTH IMPORTS (GTrack2DImport.cpp) -------------------------

// Number of kids to split num_tasks tasks between: 1 when multitasking is off
int import_num_kids(uint64_t num_tasks)
{
    if (!g_pymisha->multitasking_avail())
        return 1;

    uint64_t num_cores = max(1, (int)sysconf(_SC_NPROCESSORS_ONLN));
    uint64_t max_kids = min(min(MAX_PROCESSES2CORE * num_cores, (uint64_t)g_pymisha->max_processes()), (uint64_t)MAX_KIDS);

    return (int)max(min(max_kids, num_tasks), (uint64_t)1);
}

// Splits the input files into at most num_kids contiguous blocks balanced by size.
// Block k is files [bounds[k], bounds[k + 1]); the number of blocks is bounds.size() - 1.
vector<int> split_files(const vector<int64_t> &sizes, int num_kids)
{
    int num_files = sizes.size();
    uint64_t total = 0;
    uint64_t cum = 0;
    vector<int> bounds(1, 0);

    // +1 per file: an empty file still costs an open
    for (int64_t size : sizes)
        total += size + 1;

    // A block ends once it reaches its share of the total, or when the files left are just enough
    // to give each remaining kid one.
    for (int i = 0; i < num_files - 1; ++i) {
        int num_blocks = bounds.size();

        cum += sizes[i] + 1;
        if (num_blocks < num_kids && (cum * num_kids >= total * num_blocks || num_files - 1 - i == num_kids - num_blocks))
            bounds.push_back(i + 1);
    }
    bounds.push_back(num_files);
    return bounds;
}

// Intermediate file of a chromosome pair written by a kid in stage 1
string pair_file_name(const string &dirname, int chromid1, int chromid2, int kid)
{
    char buf[100];

    snprintf(buf, sizeof(buf), "/.%d-%d.%d", chromid1, chromid2, kid);
    return dirname + buf;
}

// Collects the intermediate files that stage 1 left in dirname, ordered by chromosome pair
vector<PairFiles> list_pair_files(const string &dirname)
{
    map<Chrom_pair, map<int, string>> pair_kid_files;
    DIR *dir = opendir(dirname.c_str());
    struct dirent *dirp;

    if (!dir)
        verror("Failed to read directory %s: %s\n", dirname.c_str(), strerror(errno));

    while ((dirp = readdir(dir))) {
        int chromid1, chromid2, kid, len = 0;

        if (sscanf(dirp->d_name, ".%d-%d.%d%n", &chromid1, &chromid2, &kid, &len) == 3 && !dirp->d_name[len])
            pair_kid_files[Chrom_pair(chromid1, chromid2)][kid] = dirname + "/" + dirp->d_name;
    }
    closedir(dir);

    vector<PairFiles> pairs;

    for (const auto &ipair : pair_kid_files) {
        PairFiles pair_files;

        pair_files.chromid1 = ipair.first.first;
        pair_files.chromid2 = ipair.first.second;
        pair_files.size = 0;
        for (const auto &ifile : ipair.second) {
            pair_files.size += BufferedFile::file_size(ifile.second.c_str());
            pair_files.files.push_back(ifile.second);
        }
        pairs.push_back(pair_files);
    }
    return pairs;
}

// The chromosome names a kid of stage 1 read: the ids that matched, and the names that the database does not have
struct KidChroms {
    vector<char>   matched;  // by chromosome id
    set<string>    seen;
    vector<string> names;

    // Notes a chromosome name of the input and what it resolved to (a negative id if nothing)
    void note(int chromid, const string &chrom) {
        if (chromid < 0)
            record(chrom);
        else {
            if ((size_t)chromid >= matched.size())
                matched.resize(chromid + 1);
            matched[chromid] = 1;
        }
    }

    void record(const string &chrom) {
        if (seen.count(chrom) || (names.size() > (size_t)UnknownChroms::MAX_TRACKED && !UnknownChroms::is_primary_chrom_name(chrom)))
            return;
        seen.insert(chrom);
        names.push_back(chrom);
    }

    // The file holds a line of the matched chromosome ids, then a line per unresolved name
    void save(const string &dirname, int kid) const;
};

string unknown_chroms_file_name(const string &dirname, int kid)
{
    return dirname + "/.unknown_chroms." + to_string(kid);
}

void KidChroms::save(const string &dirname, int kid) const
{
    string filename = unknown_chroms_file_name(dirname, kid);
    FILE *fp = fopen(filename.c_str(), "w");

    if (!fp)
        verror("Writing an intermediate file %s: %s\n", filename.c_str(), strerror(errno));
    for (size_t chromid = 0; chromid < matched.size(); ++chromid) {
        if (matched[chromid])
            fprintf(fp, "%zu ", chromid);
    }
    fprintf(fp, "\n");
    for (const string &name : names)
        fprintf(fp, "%s\n", name.c_str());
    bool failed = ferror(fp);

    if (fclose(fp) || failed)
        verror("Writing an intermediate file %s: %s\n", filename.c_str(), strerror(errno));
}

// Merges what the kids of stage 1 saved into matched (chromosome ids) and unknown, and removes their files
void load_kid_chroms(const string &dirname, int num_kids, set<int> &matched, UnknownChroms &unknown)
{
    for (int kid = 0; kid < num_kids; ++kid) {
        string filename = unknown_chroms_file_name(dirname, kid);
        ifstream file(filename);
        string name;

        if (!file || !getline(file, name))
            verror("Reading an intermediate file %s: %s\n", filename.c_str(), strerror(errno));
        istringstream ids(name);
        for (int chromid; ids >> chromid; )
            matched.insert(chromid);
        while (getline(file, name))
            unknown.record(name);
        if (file.bad())
            verror("Reading an intermediate file %s: %s\n", filename.c_str(), strerror(errno));
        file.close();
        unlink(filename.c_str());
    }
}

// Number of distinct chromosomes in the chromosome pairs: those that got data
uint64_t pairs_num_chroms(const vector<PairFiles> &pairs)
{
    set<int> chromids;

    for (const PairFiles &pair_files : pairs) {
        chromids.insert(pair_files.chromid1);
        chromids.insert(pair_files.chromid2);
    }
    return chromids.size();
}

// What an import returns to the caller for its checks and its report: the names of the input that the database
// does not have (UnknownChroms), the number of chromosomes of the input that it has, the number of chromosomes that
// got data and the number of chromosome pairs written
PyObject *import_chroms_result(const UnknownChroms &unknown, uint64_t num_matched, const vector<PairFiles> &pairs)
{
    auto string_list = [](const vector<string> &strs) -> PyObject * {
        PyObject *list = PyList_New(strs.size());
        if (!list)
            return nullptr;
        for (size_t i = 0; i < strs.size(); ++i) {
            PyObject *s = PyUnicode_FromStringAndSize(strs[i].data(), strs[i].size());
            if (!s) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, i, s);
        }
        return list;
    };

    PMPY result(PyDict_New(), true);
    PMPY py_unknown(string_list(unknown.names()), true);
    PMPY py_primary(string_list(unknown.primary_names()), true);
    PMPY py_num_unknown(PyLong_FromUnsignedLongLong(unknown.num()), true);
    PMPY py_num_primary(PyLong_FromUnsignedLongLong(unknown.num_primary()), true);
    PMPY py_truncated(PyBool_FromLong(unknown.truncated()), true);
    PMPY py_num_matched(PyLong_FromUnsignedLongLong(num_matched), true);
    PMPY py_num_data_chroms(PyLong_FromUnsignedLongLong(pairs_num_chroms(pairs)), true);
    PMPY py_num_pairs(PyLong_FromUnsignedLongLong(pairs.size()), true);

    if (!result || !py_unknown || !py_primary || !py_num_unknown || !py_num_primary || !py_truncated ||
        !py_num_matched || !py_num_data_chroms || !py_num_pairs)
        verror("Out of memory");

    PyDict_SetItemString(result, "unknown", py_unknown);
    PyDict_SetItemString(result, "num_unknown", py_num_unknown);
    PyDict_SetItemString(result, "truncated", py_truncated);
    PyDict_SetItemString(result, "unknown_primary", py_primary);
    PyDict_SetItemString(result, "num_primary", py_num_primary);
    PyDict_SetItemString(result, "num_matched", py_num_matched);
    PyDict_SetItemString(result, "num_data_chroms", py_num_data_chroms);
    PyDict_SetItemString(result, "num_pairs", py_num_pairs);
    result.to_be_stolen();
    return result;
}

// Anonymous shared memory, zeroed, that forked kids inherit. Unmapped by the parent when it goes out of scope
// (a kid ends without unwinding).
struct SharedMem {
    void  *addr;
    size_t size;

    SharedMem(size_t _size) : size(_size) {
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED)
            verror("Failed to allocate shared memory: %s", strerror(errno));
    }

    ~SharedMem() { munmap(addr, size); }
};

// Runs work(kid) for each kid - in forked kids when num_kids > 1, in this process otherwise - and returns the
// byte each work() returned. An error in a kid ends it and is raised in the parent by wait_for_kids().
vector<char> run_kids(int num_kids, const function<char(int)> &work)
{
    vector<char> res(num_kids);

    if (num_kids == 1) {
        res[0] = work(0);
        return res;
    }

    SharedMem shm(num_kids);
    char *kids_res = (char *)shm.addr;

    PyMisha::prepare4multitasking(num_kids);

    for (int kid = 0; kid < num_kids; ++kid) {
        if (!PyMisha::launch_process()) {
            kids_res[kid] = work(kid);
            _exit(0);
        }
    }

    while (PyMisha::wait_for_kids(1))
        ;

    for (int kid = 0; kid < num_kids; ++kid)
        res[kid] = kids_res[kid];
    return res;
}

// Memory of building a pair beyond its records: file buffers and the serializer (measured at 3.1 - 3.4 MB)
const uint64_t PAIR_MEM_OVERHEAD = 4 << 20;

// How long a stage 3 kid waits before it looks again for a pair that fits into the memory budget
const int64_t PAIR_WAIT_MSEC = 20;

// Estimated peak memory in bytes of building a chromosome pair whose largest subtree holds num_records records,
// bytes_per_record each. The serializer holds one subtree at a time.
uint64_t pair_mem(uint64_t num_records, uint64_t bytes_per_record)
{
    return PAIR_MEM_OVERHEAD + num_records * bytes_per_record;
}

// Number of subtrees of the quad tree of a chromosome pair of num_records records: num_records / gmax.data.size
// rounded down to a power of 4, at least 1; a gmax.data.size below 1 counts as 1.
int64_t pair_num_subtrees(const ImportArgs &args, int64_t num_records)
{
    int64_t num_subtrees = max(num_records / max((int64_t)args.max_data_size, (int64_t)1), (int64_t)1);

    return (int64_t)1 << (2 * (int)(log2(num_subtrees) / 2));  // round the number of subtrees to the lowest power of 4
}

// Name of the file that holds the records of subtree isubtree of a chromosome pair (stage 2)
string subtree_file_name(const string &dirname, int chromid1, int chromid2, int64_t isubtree)
{
    char buf[FILENAME_MAX];

    snprintf(buf, sizeof(buf), "%s/.%d-%d.s%ld", dirname.c_str(), chromid1, chromid2, (long)isubtree);
    return buf;
}

// Stage 2: runs split(pair) for every pair in num_kids kids (see run_kids), largest first, and returns what
// each call returned, in the order of pairs
vector<uint64_t> split_pairs(const vector<PairFiles> &pairs, int num_kids, const function<uint64_t(const PairFiles &)> &split)
{
    int num_pairs = pairs.size();
    vector<int> order(num_pairs);

    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int a, int b) { return pairs[a].size > pairs[b].size; });

    // int64_t next; uint64_t res[num_pairs]
    SharedMem shm((1 + num_pairs) * sizeof(int64_t));
    int64_t *next = (int64_t *)shm.addr;
    uint64_t *res = (uint64_t *)(next + 1);

    run_kids(num_kids, [&](int) {
        for (int64_t i; (i = __atomic_fetch_add(next, 1, __ATOMIC_RELAXED)) < num_pairs; )
            res[order[i]] = split(pairs[order[i]]);
        return (char)0;
    });

    return vector<uint64_t>(res, res + num_pairs);
}

// Stage 3 state that the kids share, in shared memory, read and written under lock.
// Pairs are numbered in build order: by estimated memory, largest first.
struct PairQueue {
    char     lock;
    uint64_t reserved;      // estimated memory of the pairs being built
    // followed by int next[num_pairs + 1]: next[i] == i if pair i is not taken yet, otherwise a pair to look at
    // after it; next[num_pairs] == num_pairs
};

void lock_queue(PairQueue *queue)
{
    while (__atomic_test_and_set(&queue->lock, __ATOMIC_ACQUIRE))
        sched_yield();
}

void unlock_queue(PairQueue *queue)
{
    __atomic_clear(&queue->lock, __ATOMIC_RELEASE);
}

// Returns the first pair at or after i that is not taken yet, or num_pairs
int first_untaken(int *next, int i)
{
    while (next[i] != i) {
        next[i] = next[next[i]];
        i = next[i];
    }
    return i;
}

// Stage 3: runs build(pair) for every pair in num_kids kids (see run_kids). Each kid takes the next pair,
// largest first, and a pair starts only when its estimated memory mem[pair] fits into gmax.mem.usage
// together with the estimates of the pairs being built. A smaller pair that fits may start while a larger
// one waits. A pair whose estimate exceeds gmax.mem.usage is built alone.
void build_pairs(const ImportArgs &args, const vector<PairFiles> &pairs, const vector<uint64_t> &mem, int num_kids,
                 const function<void(const PairFiles &)> &build)
{
    int num_pairs = pairs.size();
    uint64_t budget = max(args.max_mem_usage, (uint64_t)1);
    vector<int> order(num_pairs);
    vector<uint64_t> cost(num_pairs);   // estimated memory in build order, capped at the budget

    iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int a, int b) { return mem[a] > mem[b]; });
    for (int i = 0; i < num_pairs; ++i)
        cost[i] = min(mem[order[i]], budget);

    // shared memory starts zeroed: the lock is free and nothing is reserved
    SharedMem shm(sizeof(PairQueue) + (num_pairs + 1) * sizeof(int));
    PairQueue *queue = (PairQueue *)shm.addr;
    int *next = (int *)(queue + 1);

    iota(next, next + num_pairs + 1, 0);

    run_kids(num_kids, [&](int) {
        while (1) {
            bool all_taken;
            int ipair;

            // Take the first pair in build order that fits into what the running pairs leave of the budget.
            // cost is sorted from high to low, so these are the pairs from lower_bound on. A pair above the
            // budget costs the whole budget and so fits only when nothing is running.
            lock_queue(queue);
            ipair = lower_bound(cost.begin(), cost.end(), budget - queue->reserved, greater<uint64_t>()) - cost.begin();
            ipair = first_untaken(next, ipair);
            all_taken = first_untaken(next, 0) == num_pairs;
            if (ipair < num_pairs) {
                next[ipair] = ipair + 1;
                queue->reserved += cost[ipair];
            }
            unlock_queue(queue);

            if (all_taken)
                break;

            if (ipair == num_pairs) {
                // nothing fits now: wait for a running pair to end
                struct timespec req;

                set_rel_timeout(PAIR_WAIT_MSEC, req);
                nanosleep(&req, NULL);
                check_interrupt();
                continue;
            }

            build(pairs[order[ipair]]);

            lock_queue(queue);
            queue->reserved -= cost[ipair];
            unlock_queue(queue);
        }
        return (char)0;
    });
}

// Opens an input file for stage 1 and reads its header: the first 6 columns must be named as GInterval2D's,
// followed by the value. Returns false for an empty file.
bool open_intervals_file(BufferedFile &infile, const string &fname, vector<string> &fields, long &lineno)
{
    if (infile.open(fname.c_str(), "r"))
        verror("Failed to open file %s: %s", fname.c_str(), strerror(errno));

    lineno += split_line(infile, fields, '\t');

    if (fields.empty())
        return false;

    if (fields.size() < NUM_2D_COLS + 1)
        verror("File %s, line %ld: invalid format", infile.file_name().c_str(), lineno);

    for (int i = 0; i < NUM_2D_COLS; ++i) {
        if (fields[i] != COL_NAMES_2D[i])
            verror("File %s, line %ld: invalid format", infile.file_name().c_str(), lineno);
    }
    return true;
}

// Reads the next interval of an input file and notes its chromosome names in chroms. Returns 0 at EOF, -1 for a
// line on a chromosome the database does not have (skipped, as R skips it), 1 otherwise.
int read_interval_line(const ImportArgs &args, BufferedFile &infile, vector<string> &fields, long &lineno, KidChroms &chroms,
                       int &chromid1, int &chromid2, int64_t &start1, int64_t &end1, int64_t &start2, int64_t &end2, float &val)
{
    char *endptr;

    lineno += split_line(infile, fields, '\t');
    check_interrupt();

    if (fields.empty())
        return 0;

    if (fields.size() < NUM_2D_COLS + 1)
        verror("File %s, line %ld: invalid format", infile.file_name().c_str(), lineno);

    // there might be unrecognized chromosomes: skip them, and report them once the track is written
    chromid1 = args.chromkey->find_chromid(fields[CHROM1]);
    chromid2 = args.chromkey->find_chromid(fields[CHROM2]);
    chroms.note(chromid1, fields[CHROM1]);
    chroms.note(chromid2, fields[CHROM2]);
    if (chromid1 < 0 || chromid2 < 0)
        return -1;

    start1 = strtoll(fields[START1].c_str(), &endptr, 10);
    if (*endptr || start1 < 0)
        verror("File %s, line %ld: invalid format of start1 coordinate", infile.file_name().c_str(), lineno);

    end1 = strtoll(fields[END1].c_str(), &endptr, 10);
    if (*endptr || end1 < 0)
        verror("File %s, line %ld: invalid format of end1 coordinate", infile.file_name().c_str(), lineno);

    if (start1 >= end1)
        verror("File %s, line %ld: start1 coordinate exceeds or equals the end1 coordinate", infile.file_name().c_str(), lineno);

    if ((uint64_t)end1 > args.chromkey->get_chrom_size(chromid1))
        verror("File %s, line %ld: end1 coordinate exceeds chromosome's size", infile.file_name().c_str(), lineno);

    start2 = strtoll(fields[START2].c_str(), &endptr, 10);
    if (*endptr || start2 < 0)
        verror("File %s, line %ld: invalid format of start2 coordinate", infile.file_name().c_str(), lineno);

    end2 = strtoll(fields[END2].c_str(), &endptr, 10);
    if (*endptr || end2 < 0)
        verror("File %s, line %ld: invalid format of end2 coordinate", infile.file_name().c_str(), lineno);

    if (start2 >= end2)
        verror("File %s, line %ld: start2 coordinate exceeds or equals the end2 coordinate", infile.file_name().c_str(), lineno);

    if ((uint64_t)end2 > args.chromkey->get_chrom_size(chromid2))
        verror("File %s, line %ld: end2 coordinate exceeds chromosome's size", infile.file_name().c_str(), lineno);

    val = strtod(fields[NUM_2D_COLS].c_str(), &endptr);
    if (*endptr)
        verror("File %s, line %ld: invalid value", infile.file_name().c_str(), lineno);

    return 1;
}

// Intermediate files of a kid in stage 1, one per chromosome pair
class PairOutFiles : public unordered_map<Chrom_pair, BufferedFile *> {
public:
    ~PairOutFiles() {
        for (iterator ifile = begin(); ifile != end(); ++ifile)
            delete ifile->second;
    }

    BufferedFile &get(const string &dirname, int chromid1, int chromid2, int kid) {
        iterator ifile = find(Chrom_pair(chromid1, chromid2));

        if (ifile == end()) {
            string filename = pair_file_name(dirname, chromid1, chromid2, kid);

            ifile = insert(make_pair(Chrom_pair(chromid1, chromid2), new BufferedFile())).first;
            if (ifile->second->open(filename.c_str(), "wb"))
                verror("Writing an intermediate file %s: %s\n", filename.c_str(), strerror(errno));
        }
        return *ifile->second;
    }

    // Closes the files and checks: a kid ends without unwinding, and the last buffered writes can still fail.
    // It also keeps the number of open files low for the next stage.
    void close() {
        for (iterator ifile = begin(); ifile != end(); ++ifile) {
            if (ifile->second->close())
                verror("Writing an intermediate file %s: %s\n", ifile->second->file_name().c_str(), strerror(errno));
        }
    }
};

//------------------------------- gtrack_2d_import (GTrack2DImport.cpp) -------------------------------

const int64_t INTERVAL_RECORD_SIZE = 4 * sizeof(int64_t) + sizeof(float);

// Peak memory of building a chromosome pair, in bytes per record (see pair_mem): the quad tree's objects
// (24 bytes a point, 40 a rectangle), object pointers and nodes, as the vectors holding them grow. Measured by R
// misha as the peak RSS of a serial import of one pair of 2M - 40M Hi-C-like records: at most 74.8 bytes a point
// and 98.4 a rectangle. Rounded up to a multiple of 8.
const uint64_t POINT_MEM = 80;
const uint64_t RECT_MEM = 104;

bool read_interval(BufferedFile &f, int64_t &start1, int64_t &end1, int64_t &start2, int64_t &end2, float &val)
{
    f.read(&start1, sizeof(start1));
    f.read(&end1, sizeof(end1));
    f.read(&start2, sizeof(start2));
    f.read(&end2, sizeof(end2));
    f.read(&val, sizeof(val));

    if (f.eof())
        return false;

    if (f.error())
        verror("Reading file %s: %s\n", f.file_name().c_str(), strerror(errno));

    return true;
}

void write_interval(BufferedFile &f, int64_t start1, int64_t end1, int64_t start2, int64_t end2, float val)
{
    f.write(&start1, sizeof(start1));
    f.write(&end1, sizeof(end1));
    f.write(&start2, sizeof(start2));
    f.write(&end2, sizeof(end2));
    f.write(&val, sizeof(val));
    if (f.error())
        verror("Writing file %s: %s\n", f.file_name().c_str(), strerror(errno));
}

// Calls f on every interval of the intermediate files, in order, and deletes each file once it is read
template <typename F>
void for_each_interval(const vector<string> &files, F f)
{
    int64_t start1, end1, start2, end2;
    float val;

    for (const string &fname : files) {
        BufferedFile file;

        if (file.open(fname.c_str(), "r"))
            verror("Opening an intermediate file %s: %s\n", fname.c_str(), strerror(errno));

        while (read_interval(file, start1, end1, start2, end2, val)) {
            f(start1, end1, start2, end2, val);
            check_interrupt();
        }
        file.close();
        unlink(fname.c_str());
    }
}

// STAGE 1 of one kid: reads files [fbegin, fend) into the kid's intermediate files, one per chromosome pair.
// Returns true if all the intervals are points.
bool read_input_files(const ImportArgs &args, int fbegin, int fend, int kid)
{
    PairOutFiles out_files;
    KidChroms chroms;
    bool are_all_points = true;
    int64_t start1, start2, end1, end2;
    float val;

    for (int ifile = fbegin; ifile < fend; ++ifile) {
        vector<string> fields;
        long lineno = 0;
        BufferedFile infile;
        int chromid1, chromid2, res;

        if (!open_intervals_file(infile, args.files[ifile], fields, lineno))
            continue;

        while ((res = read_interval_line(args, infile, fields, lineno, chroms, chromid1, chromid2, start1, end1, start2, end2, val))) {
            if (res < 0)
                continue;

            are_all_points &= start1 == end1 - 1 && start2 == end2 - 1;
            write_interval(out_files.get(args.dirname, chromid1, chromid2, kid), start1, end1, start2, end2, val);
        }
    }

    out_files.close();
    chroms.save(args.dirname, kid);
    return are_all_points;
}

// STAGE 2 for a chromosome pair whose quad tree has several subtrees: splits its intermediate files into one
// file per subtree. Returns the number of records of the largest subtree plus the number of rectangles that cross
// a subtree border, which the serializer keeps until the pair ends: what its memory is estimated by.
uint64_t split_pair(const ImportArgs &args, const PairFiles &pair_files)
{
    int chromid1 = pair_files.chromid1;
    int chromid2 = pair_files.chromid2;
    int64_t num_subtrees = pair_num_subtrees(args, pair_files.size / INTERVAL_RECORD_SIZE);
    Rectangles subarenas = RectsQuadTreeCachedSerializer::split_arena(0, 0, args.chromkey->get_chrom_size(chromid1),
                                                                       args.chromkey->get_chrom_size(chromid2), num_subtrees);
    BufferedFiles subtrees_files(num_subtrees);
    vector<uint64_t> num_records(num_subtrees, 0);
    uint64_t num_border = 0;

    for (uint64_t i = 0; i < subtrees_files.size(); ++i) {
        string fname = subtree_file_name(args.dirname, chromid1, chromid2, i);

        subtrees_files[i] = new BufferedFile();
        if (subtrees_files[i]->open(fname.c_str(), "w"))
            verror("Opening an intermediate file %s: %s\n", fname.c_str(), strerror(errno));
    }

    for_each_interval(pair_files.files, [&](int64_t start1, int64_t end1, int64_t start2, int64_t end2, float val) {
        Rectangle rect(start1, start2, end1, end2);
        for (Rectangles::const_iterator isubarena = subarenas.begin(); isubarena != subarenas.end(); ++isubarena) {
            if (rect.do_intersect(*isubarena)) {
                write_interval(*subtrees_files[isubarena - subarenas.begin()], start1, end1, start2, end2, val);
                num_records[isubarena - subarenas.begin()]++;
                num_border += !rect.is_inside(*isubarena);
                break;
            }
        }
    });

    for (uint64_t i = 0; i < subtrees_files.size(); ++i) {
        if (subtrees_files[i]->close())
            verror("Writing an intermediate file %s: %s\n", subtrees_files[i]->file_name().c_str(), strerror(errno));
    }

    return *max_element(num_records.begin(), num_records.end()) + num_border;
}

// The files to read for each subtree of a pair: its intermediate files, or its subtree files if split_pair split it
vector<vector<string>> subtree_sources(const ImportArgs &args, const PairFiles &pair_files, int64_t num_subtrees)
{
    vector<vector<string>> subtree_srcs(1, pair_files.files);

    if (num_subtrees > 1) {
        subtree_srcs.clear();
        for (int64_t i = 0; i < num_subtrees; ++i)
            subtree_srcs.push_back(vector<string>(1, subtree_file_name(args.dirname, pair_files.chromid1, pair_files.chromid2, i)));
    }
    return subtree_srcs;
}

string pair_track_file(const ImportArgs &args, int chromid1, int chromid2)
{
    return args.dirname + "/" + GenomeTrack::get_2d_filename(*args.chromkey, chromid1, chromid2);
}

// STAGE 3 for one chromosome pair: builds its quad tree from its intermediate files, or from its subtree files
// if split_pair split it.
void write_pair(const ImportArgs &args, const PairFiles &pair_files, bool are_all_points)
{
    int chromid1 = pair_files.chromid1;
    int chromid2 = pair_files.chromid2;
    int64_t size1 = args.chromkey->get_chrom_size(chromid1);
    int64_t size2 = args.chromkey->get_chrom_size(chromid2);
    int64_t num_subtrees = pair_num_subtrees(args, pair_files.size / INTERVAL_RECORD_SIZE);
    PointsQuadTreeCachedSerializer points_serializer;
    RectsQuadTreeCachedSerializer rects_serializer;
    BufferedFile file;
    string filename = pair_track_file(args, chromid1, chromid2);

    if (are_all_points) {
        write_signature(file, filename, POINTS_SIGNATURE);
        points_serializer.begin(file, 0, 0, size1, size2, (unsigned)num_subtrees, true, TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);
    } else {
        write_signature(file, filename, RECTS_SIGNATURE);
        rects_serializer.begin(file, 0, 0, size1, size2, (unsigned)num_subtrees, true, TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);
    }

    // Read the records of a subtree and insert them to StatQuadTreeCachedSerializer.
    for (const vector<string> &srcs : subtree_sources(args, pair_files, num_subtrees)) {
        for_each_interval(srcs, [&](int64_t start1, int64_t end1, int64_t start2, int64_t end2, float val) {
            if (are_all_points)
                points_serializer.insert(PointsQuadTree::ValueType(start1, start2, val));
            else
                rects_serializer.insert(RectsQuadTree::ValueType(start1, start2, end1, end2, val));
        });
    }

    if (are_all_points)
        points_serializer.end();
    else
        rects_serializer.end();
    close_chk(file);
}

//------------------------------- gtrack_2d_import_contacts (GTrack2DImportContacts.cpp) -------------

const int64_t CONTACT_RECORD_SIZE = 2 * sizeof(int64_t) + sizeof(float);

// Peak memory of building a chromosome pair, in bytes per contact (see pair_mem): the sorted vector of the
// records that sums the duplicates plus the quad trees, which hold every contact twice (mirrored in a cis pair,
// one tree per orientation in a trans pair). Measured by R misha (5.15.1) as the peak RSS of a serial import of
// one pair of 2M - 40M Hi-C-like contacts: at most 167.3 bytes a contact (cis, no duplicates; trans and
// duplicates need less). Rounded up to a multiple of 8.
const uint64_t CONTACT_MEM = 168;

unsigned read_header(BufferedFile &file, const char *fname, const char *ftype, const char *colnames[], int num_cols, vector<int> &fcol_idx)
{
    vector<string> fields;

    if (file.open(fname, "r"))
        verror("Failed to open %s file %s: %s", ftype, fname, strerror(errno));

    fcol_idx.resize(num_cols, -1);

    split_line(file, fields, '\t');
    if (fields.empty())
        verror("Invalid format of %s file %s", ftype, fname);

    if (file.error())
        verror("Error while reading %s file %s: %s", ftype, fname, strerror(errno));

    for (int i = 0; i < num_cols; ++i) {
        for (vector<string>::const_iterator ifield = fields.begin(); ifield != fields.end(); ++ifield) {
            if (*ifield == colnames[i]) {
                if (fcol_idx[i] == -1)
                    fcol_idx[i] = ifield - fields.begin();
                else
                    verror("Invalid format of %s file %s: column %s appears more than once", ftype, fname, colnames[i]);
            }
        }

        if (fcol_idx[i] < 0)
            verror("Invalid format of %s file %s: column %s was not found", ftype, fname, colnames[i]);
    }

    return fields.size();
}

void write_contact(BufferedFile &file, int64_t coord1, int64_t coord2, float value)
{
    file.write(&coord1, sizeof(coord1));
    file.write(&coord2, sizeof(coord2));
    file.write(&value, sizeof(value));
    if (file.error())
        verror("Writing file %s: %s\n", file.file_name().c_str(), strerror(errno));
}

// Calls f on every contact of the intermediate files, in order, and deletes each file once it is read
template <typename F>
void for_each_contact(const vector<string> &files, F f)
{
    int64_t coord1, coord2;
    float value;

    for (const string &fname : files) {
        BufferedFile file;

        if (file.open(fname.c_str(), "r"))
            verror("Opening an intermediate file %s: %s\n", fname.c_str(), strerror(errno));

        while (1) {
            file.read(&coord1, sizeof(coord1));
            file.read(&coord2, sizeof(coord2));
            file.read(&value, sizeof(value));

            if (file.eof())
                break;

            if (file.error())
                verror("Reading file %s: %s\n", file.file_name().c_str(), strerror(errno));

            f(coord1, coord2, value);
            check_interrupt();
        }
        file.close();
        unlink(fname.c_str());
    }
}

// STAGE 1, intervals-value format: each kid reads a contiguous block of the files. unknown receives the chromosome
// names of the files that the database does not have. Returns the number of chromosome names of the files that the
// database has.
uint64_t process_contacts_as_intervals(const ImportArgs &args, UnknownChroms &unknown)
{
    vector<int64_t> sizes = get_files_sizes(args.files);
    vector<int> bounds = split_files(sizes, import_num_kids(sizes.size()));

    run_kids(bounds.size() - 1, [&](int kid) {
        PairOutFiles out_files;
        KidChroms chroms;
        int64_t start1, start2, end1, end2;
        float val;

        for (int ifile = bounds[kid]; ifile < bounds[kid + 1]; ++ifile) {
            vector<string> fields;
            long lineno = 0;
            BufferedFile infile;
            int chromid1, chromid2, res;

            if (!open_intervals_file(infile, args.files[ifile], fields, lineno))
                continue;

            while ((res = read_interval_line(args, infile, fields, lineno, chroms, chromid1, chromid2, start1, end1, start2, end2, val))) {
                if (res < 0)
                    continue;

                int64_t coord1 = (start1 + end1) / 2;
                int64_t coord2 = (start2 + end2) / 2;

                if (chromid1 > chromid2 || (chromid1 == chromid2 && coord1 > coord2)) {
                    swap(chromid1, chromid2);
                    swap(coord1, coord2);
                }

                write_contact(out_files.get(args.dirname, chromid1, chromid2, kid), coord1, coord2, val);
            }
        }
        out_files.close();
        chroms.save(args.dirname, kid);
        return (char)0;
    });

    set<int> matched;
    load_kid_chroms(args.dirname, bounds.size() - 1, matched, unknown);
    return matched.size();
}

// The largest fend id accepted
const int64_t MAX_FEND_ID = INT32_MAX;

// R misha's FendIndex (GTrack2DImportContacts.cpp). The chromosome and coordinate of each fend, by fend
// id. Held in arrays indexed by the id (12 bytes an id) while the ids are dense enough, as real fends
// files number their fends 1..N; once an id would make the arrays too sparse for the number of fends
// (ids near 1e9 would take 12 GB), in a hash table, and back in arrays as soon as the fends added since
// make the ids dense again (a file not sorted by id can start with large ids). The two thresholds differ
// by a factor of 2, so that the number of fends at least doubles between two switches: each switch
// copies every fend.
class FendIndex {
public:
    enum { NOT_LISTED = -2, UNKNOWN_CHROM = -1 };

    // Adds a fend; returns false if the id is already there
    bool add(int64_t id, int chromid, int64_t coord)
    {
        ++m_num_fends;
        m_max_id = max(m_max_id, id);
        if (m_sparse) {
            bool added = m_map.emplace(id, Fend{ chromid, coord }).second;
            if ((uint64_t)m_max_id < max(MIN_DENSE_IDS, DENSE_IDS_PER_FEND * m_num_fends))
                to_dense();
            return added;
        }

        if ((uint64_t)id >= m_chromids.size()) {
            if ((uint64_t)id >= max(MIN_DENSE_IDS, MAX_IDS_PER_FEND * m_num_fends)) {
                to_sparse();
                return m_map.emplace(id, Fend{ chromid, coord }).second;
            }
            // grow only: the ids need not be sorted, and shrinking would drop the fends read so far
            m_chromids.resize(id + 1, NOT_LISTED);
            m_coords.resize(id + 1);
        }
        if (m_chromids[id] != NOT_LISTED)
            return false;
        m_chromids[id] = chromid;
        m_coords[id] = coord;
        return true;
    }

    // The chromosome of the fend (NOT_LISTED if the file does not list it) and its coordinate
    int chromid(int64_t id) const
    {
        if (m_sparse) {
            auto it = m_map.find(id);
            return it == m_map.end() ? NOT_LISTED : it->second.chromid;
        }
        return id >= 0 && (uint64_t)id < m_chromids.size() ? m_chromids[id] : NOT_LISTED;
    }

    int64_t coord(int64_t id) const { return m_sparse ? m_map.find(id)->second.coord : m_coords[id]; }

private:
    struct Fend {
        int     chromid;
        int64_t coord;
    };

    // Arrays of MIN_DENSE_IDS ids are kept whatever the number of fends. Past that, an id of
    // MAX_IDS_PER_FEND ids a fend or more moves the fends to the hash table, and they move back once the
    // largest id is below DENSE_IDS_PER_FEND ids a fend.
    static constexpr uint64_t MIN_DENSE_IDS = 1 << 24;
    static constexpr uint64_t MAX_IDS_PER_FEND = 8;
    static constexpr uint64_t DENSE_IDS_PER_FEND = 4;

    bool                              m_sparse{false};
    uint64_t                          m_num_fends{0};
    int64_t                           m_max_id{-1};
    vector<int>                       m_chromids;
    vector<int64_t>                   m_coords;
    unordered_map<int64_t, Fend>      m_map;

    void to_sparse()
    {
        m_map.reserve(m_num_fends);
        for (size_t id = 0; id < m_chromids.size(); ++id) {
            if (m_chromids[id] != NOT_LISTED)
                m_map.emplace(id, Fend{ m_chromids[id], m_coords[id] });
        }
        vector<int>().swap(m_chromids);
        vector<int64_t>().swap(m_coords);
        m_sparse = true;
    }

    void to_dense()
    {
        m_chromids.assign(m_max_id + 1, NOT_LISTED);
        m_coords.assign(m_max_id + 1, 0);
        for (const auto &fend : m_map) {
            m_chromids[fend.first] = fend.second.chromid;
            m_coords[fend.first] = fend.second.coord;
        }
        unordered_map<int64_t, Fend>().swap(m_map);
        m_sparse = false;
    }
};

// STAGE 1, fends-value format: the fends file is read here, then each kid reads a contiguous block of the
// contacts files. unknown receives the chromosome names of the fends file that the database does not have; the
// contacts of their fends are skipped. Returns the number of chromosomes of the fends file that the database has.
uint64_t process_contacts_as_fends(const ImportArgs &args, const char *fends_fname, UnknownChroms &unknown)
{
    enum { COL_FEND, COL_CHR, COL_COORD, NUM_FENDS_COLS };
    const char *FENDS_COLS[] = { "fend", "chr", "coord" };
    vector<int> fends_col_idx;
    unsigned fends_cols_total;

    enum { COL_FEND1, COL_FEND2, COL_COUNT, NUM_CONTACTS_COLS };
    const char *CONTACTS_COLS[] = { "fend1", "fend2", "count" };

    vector<string> fields;
    vector<int64_t> sizes = get_files_sizes(args.files);

    BufferedFile fends_file;
    fends_cols_total = read_header(fends_file, fends_fname, "fragment ends", FENDS_COLS, NUM_FENDS_COLS, fends_col_idx);

    // chromid of a fend: FendIndex::NOT_LISTED, FendIndex::UNKNOWN_CHROM, or the chromosome of the fend
    FendIndex fends;
    set<int> matched_chromids;
    int64_t lineno;
    char *endptr;
    int64_t fendid;
    int64_t coord;

    lineno = 1;
    while (1) {
        lineno++;
        split_line(fends_file, fields, '\t');
        if (fields.empty())
            break;

        if (fields.size() != fends_cols_total)
            verror("Error reading fragment ends file %s, line %ld: invalid file format", fends_fname, (long)lineno);

        const char *fendid_str = fields[fends_col_idx[COL_FEND]].c_str();
        const char *chr_str = fields[fends_col_idx[COL_CHR]].c_str();
        const char *coord_str = fields[fends_col_idx[COL_COORD]].c_str();

        errno = 0;
        fendid = strtoll(fendid_str, &endptr, 10);
        if (*endptr || endptr == fendid_str)
            verror("Error reading fragment ends file %s, line %ld: invalid %s", fends_fname, (long)lineno, FENDS_COLS[COL_FEND]);

        if (fendid < 0 || errno == ERANGE || fendid > MAX_FEND_ID)
            verror("Error reading fragment ends file %s, line %ld: %s %s is out of range (0 - %lld)", fends_fname, (long)lineno,
                   FENDS_COLS[COL_FEND], fendid_str, (long long)MAX_FEND_ID);

        if (fends.chromid(fendid) != FendIndex::NOT_LISTED)
            verror("Error reading fragment ends file %s, line %ld: %s %lld appears more than once", fends_fname, (long)lineno,
                   FENDS_COLS[COL_FEND], (long long)fendid);

        // Use chromosome name as-is (no forced chr prefix)
        int chromid = args.chromkey->find_chromid(chr_str);
        if (chromid < 0)
            chromid = FendIndex::UNKNOWN_CHROM;

        coord = strtoll(coord_str, &endptr, 10);
        if (*endptr)
            verror("Error reading fragment ends file %s, line %ld: invalid %s", fends_fname, (long)lineno, FENDS_COLS[COL_COORD]);

        if (chromid == FendIndex::UNKNOWN_CHROM)
            unknown.record(chr_str);
        else if (coord < 0 || coord >= (int64_t)args.chromkey->get_chrom_size(chromid))
            verror("Error reading fragment ends file %s, line %ld: %s is out of range", fends_fname, (long)lineno, FENDS_COLS[COL_COORD]);
        else
            matched_chromids.insert(chromid);

        fends.add(fendid, chromid, coord);
        check_interrupt();
    }
    fends_file.close();

    vector<int> bounds = split_files(sizes, import_num_kids(sizes.size()));

    run_kids(bounds.size() - 1, [&](int kid) {
        PairOutFiles out_files;
        vector<string> fields;
        int64_t lineno;
        char *endptr;
        float value;

        for (int icontact_file = bounds[kid]; icontact_file < bounds[kid + 1]; ++icontact_file) {
            int64_t fend1, fend2;
            const char *contacts_fname = args.files[icontact_file].c_str();
            vector<int> contacts_col_idx;
            unsigned contacts_cols_total;
            BufferedFile contacts_file;

            contacts_cols_total = read_header(contacts_file, contacts_fname, "contacts", CONTACTS_COLS, NUM_CONTACTS_COLS, contacts_col_idx);

            lineno = 1;
            while (1) {
                lineno++;
                split_line(contacts_file, fields, '\t');
                if (fields.empty())
                    break;

                if (fields.size() != contacts_cols_total)
                    verror("Error reading contacts ends file %s, line %ld: invalid file format", contacts_fname, (long)lineno);

                const char *fend1_str = fields[contacts_col_idx[COL_FEND1]].c_str();
                const char *fend2_str = fields[contacts_col_idx[COL_FEND2]].c_str();
                const char *value_str = fields[contacts_col_idx[COL_COUNT]].c_str();

                fend1 = strtoll(fend1_str, &endptr, 10);
                if (*endptr)
                    verror("Error reading contacts file %s, line %ld: invalid %s", contacts_fname, (long)lineno, CONTACTS_COLS[COL_FEND1]);
                int chromid1 = fends.chromid(fend1);
                if (chromid1 < 0)
                    continue;

                fend2 = strtoll(fend2_str, &endptr, 10);
                if (*endptr)
                    verror("Error reading contacts file %s, line %ld: invalid %s", contacts_fname, (long)lineno, CONTACTS_COLS[COL_FEND2]);
                int chromid2 = fends.chromid(fend2);
                if (chromid2 < 0)
                    continue;

                value = strtod(value_str, &endptr);
                if (*endptr)
                    verror("Error reading contacts file %s, line %ld: invalid %s", contacts_fname, (long)lineno, CONTACTS_COLS[COL_COUNT]);

                int64_t coord1 = fends.coord(fend1);
                int64_t coord2 = fends.coord(fend2);
                if (chromid1 > chromid2 || (chromid1 == chromid2 && coord1 > coord2)) {
                    swap(chromid1, chromid2);
                    swap(coord1, coord2);
                }

                write_contact(out_files.get(args.dirname, chromid1, chromid2, kid), coord1, coord2, value);
                check_interrupt();
            }
        }
        out_files.close();
        return (char)0;
    });
    return matched_chromids.size();
}

// The number of subtrees of a pair's quad trees: a cis pair holds every contact twice, mirrored around the diagonal
int64_t contacts_num_subtrees(const ImportArgs &args, const PairFiles &pair_files)
{
    int64_t num_contacts = pair_files.size / CONTACT_RECORD_SIZE;

    return pair_num_subtrees(args, pair_files.chromid1 == pair_files.chromid2 ? 2 * num_contacts : num_contacts);
}

// STAGE 2 for a chromosome pair whose quad trees have several subtrees: splits its contacts into one file per
// subtree, mirrored around the diagonal in a cis pair. Returns the number of contacts of the largest subtree,
// which the memory of the pair is estimated by.
uint64_t split_contacts_pair(const ImportArgs &args, const PairFiles &pair_files)
{
    int chromid1 = pair_files.chromid1;
    int chromid2 = pair_files.chromid2;
    int64_t num_subtrees = contacts_num_subtrees(args, pair_files);
    Rectangles subarenas = PointsQuadTreeCachedSerializer::split_arena(0, 0, args.chromkey->get_chrom_size(chromid1),
                                                                        args.chromkey->get_chrom_size(chromid2), num_subtrees);
    BufferedFiles subtrees_files(num_subtrees);
    vector<uint64_t> num_contacts(num_subtrees, 0);

    for (uint64_t i = 0; i < subtrees_files.size(); ++i) {
        string fname = subtree_file_name(args.dirname, chromid1, chromid2, i);

        subtrees_files[i] = new BufferedFile();
        if (subtrees_files[i]->open(fname.c_str(), "w"))
            verror("Opening an intermediate file %s: %s\n", fname.c_str(), strerror(errno));
    }

    auto write_to_subtree = [&](int64_t coord1, int64_t coord2, float value) {
        Point point(coord1, coord2);
        for (Rectangles::const_iterator isubarena = subarenas.begin(); isubarena != subarenas.end(); ++isubarena) {
            if (point.do_intersect(*isubarena)) {
                write_contact(*subtrees_files[isubarena - subarenas.begin()], coord1, coord2, value);
                num_contacts[isubarena - subarenas.begin()]++;
                break;
            }
        }
    };

    for_each_contact(pair_files.files, [&](int64_t coord1, int64_t coord2, float value) {
        write_to_subtree(coord1, coord2, value);

        // if it's a pair of identical chromosomes, mirror the contact around the diagonal
        if (chromid1 == chromid2 && coord1 != coord2)
            write_to_subtree(coord2, coord1, value);
    });

    for (uint64_t i = 0; i < subtrees_files.size(); ++i) {
        if (subtrees_files[i]->close())
            verror("Writing an intermediate file %s: %s\n", subtrees_files[i]->file_name().c_str(), strerror(errno));
    }

    return *max_element(num_contacts.begin(), num_contacts.end());
}

// STAGE 3 for one chromosome pair: builds its quad trees from its intermediate files, or from its subtree files
// if split_contacts_pair split it.
void write_contacts_pair(const ImportArgs &args, const PairFiles &pair_files, bool allow_duplicates)
{
    int chromid1 = pair_files.chromid1;
    int chromid2 = pair_files.chromid2;
    int64_t size1 = args.chromkey->get_chrom_size(chromid1);
    int64_t size2 = args.chromkey->get_chrom_size(chromid2);
    int64_t num_subtrees = contacts_num_subtrees(args, pair_files);
    BufferedFile file1, file2;
    PointsQuadTreeCachedSerializer qtree_serializer1, qtree_serializer2;

    write_signature(file1, pair_track_file(args, chromid1, chromid2), POINTS_SIGNATURE);
    qtree_serializer1.begin(file1, 0, 0, size1, size2, (unsigned)num_subtrees, false, TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);

    if (chromid1 != chromid2) {
        write_signature(file2, pair_track_file(args, chromid2, chromid1), POINTS_SIGNATURE);
        qtree_serializer2.begin(file2, 0, 0, size2, size1, (unsigned)num_subtrees, false, TRACK_CHUNK_SIZE, TRACK_NUM_CHUNKS);
    }

    struct Contact {
        uint64_t coord1;
        uint64_t coord2;
        float value;
    };

    // Read the contacts of a subtree, sum up the duplicates and insert them to StatQuadTreeCachedSerializer
    // in coordinate order, as R misha 5.15.1 does. A hash map used to sum them, and its iteration order, which
    // depends on the C++ library, decided the bytes of the track files. Sorting needs less memory than the map did.
    for (const vector<string> &srcs : subtree_sources(args, pair_files, num_subtrees)) {
        uint64_t num_records = 0;

        for (const string &fname : srcs) {
            struct stat st;

            if (stat(fname.c_str(), &st))
                verror("Accessing an intermediate file %s: %s\n", fname.c_str(), strerror(errno));
            num_records += st.st_size / CONTACT_RECORD_SIZE;
        }

        vector<Contact> contacts;
        contacts.reserve(num_records);
        for_each_contact(srcs, [&](int64_t coord1, int64_t coord2, float value) {
            contacts.push_back(Contact{(uint64_t)coord1, (uint64_t)coord2, value});
        });

        // stable: the duplicates of a contact are summed in the order they were read
        stable_sort(contacts.begin(), contacts.end(), [](const Contact &a, const Contact &b) {
            return a.coord1 < b.coord1 || (a.coord1 == b.coord1 && a.coord2 < b.coord2);
        });

        for (size_t i = 0; i < contacts.size(); ) {
            int64_t coord1 = contacts[i].coord1;
            int64_t coord2 = contacts[i].coord2;
            double sum = contacts[i].value;
            size_t j = i + 1;

            for (; j < contacts.size() && contacts[j].coord1 == contacts[i].coord1 && contacts[j].coord2 == contacts[i].coord2; ++j)
                sum += contacts[j].value;

            if (j - i > 1 && !allow_duplicates)
                verror("Duplicated contact (%s, %ld)-(%s, %ld)", args.chromkey->id2chrom(chromid1).c_str(), (long)coord1,
                       args.chromkey->id2chrom(chromid2).c_str(), (long)coord2);

            float value = sum;

            qtree_serializer1.insert(PointsQuadTree::ValueType(coord1, coord2, value));

            if (chromid1 != chromid2)
                qtree_serializer2.insert(PointsQuadTree::ValueType(coord2, coord1, value));
            else if (num_subtrees == 1 && coord1 != coord2)
                qtree_serializer1.insert(PointsQuadTree::ValueType(coord2, coord1, value));

            i = j;
            check_interrupt();
        }

        check_interrupt();
    }

    qtree_serializer1.end();
    close_chk(file1);

    if (chromid1 != chromid2) {
        qtree_serializer2.end();
        close_chk(file2);
    }
}

// Stages 2 and 3 of both imports: split the pairs of several subtrees, then build the pairs.
// num_subtrees(pair), split(pair) and build(pair) are the import's; bytes_per_record its memory estimate.
void split_and_build(const ImportArgs &args, const vector<PairFiles> &pairs, int64_t record_size,
                     const function<int64_t(const PairFiles &)> &num_subtrees,
                     const function<uint64_t(const PairFiles &)> &split, uint64_t bytes_per_record,
                     const function<void(const PairFiles &)> &build)
{
    // the records that the memory estimate of each pair counts: all of them, or those of its largest subtree
    vector<uint64_t> part_records;

    // STAGE 2: split the pairs of several subtrees into one file per subtree. Each stage has its own PyMisha:
    // the multitasking state (shared memory, kid bookkeeping) is set up once per PyMisha.
    {
        PyMisha pymisha(true);
        vector<PairFiles> split_list;
        vector<int> split_idx;
        uint64_t num_split_records = 0;

        for (size_t ipair = 0; ipair < pairs.size(); ++ipair) {
            int64_t num_records = pairs[ipair].size / record_size;

            part_records.push_back(num_records);
            if (num_subtrees(pairs[ipair]) > 1) {
                split_list.push_back(pairs[ipair]);
                split_idx.push_back(ipair);
                num_split_records += num_records;
            }
        }

        if (!split_list.empty()) {
            int num_kids = import_num_kids(min((uint64_t)split_list.size(), num_split_records / MIN_RECORDS_PER_PROCESS));
            vector<uint64_t> res = split_pairs(split_list, num_kids, split);

            for (size_t i = 0; i < split_list.size(); ++i)
                part_records[split_idx[i]] = res[i];
        }
        check_interrupt();
    }

    // STAGE 3: build the quad trees
    {
        PyMisha pymisha(true);
        uint64_t num_records = 0;

        for (const PairFiles &pair_files : pairs)
            num_records += pair_files.size / record_size;

        int num_kids = import_num_kids(min((uint64_t)pairs.size(), num_records / MIN_RECORDS_PER_PROCESS));
        vector<uint64_t> mem;

        for (uint64_t num_part_records : part_records)
            mem.push_back(pair_mem(num_part_records, bytes_per_record));

        build_pairs(args, pairs, mem, num_kids, build);
    }
}

// Reads the arguments shared by both entry points
void read_import_args(ImportArgs &args, const char *dir_str, PyObject *files_obj, long long max_data_size, long long max_mem_usage_kb)
{
    if (!g_pmdb || !g_pmdb->is_initialized())
        verror("Database not initialized. Call gdb_init() first.");

    args.dirname = dir_str;
    args.max_data_size = max_data_size < 0 ? 0 : (uint64_t)max_data_size;
    // gmax.mem.usage is in KB
    args.max_mem_usage = max_mem_usage_kb < 0 ? (uint64_t)-1 : (uint64_t)max_mem_usage_kb * 1000;
    args.chromkey = &g_pmdb->chromkey();

    PMPY seq(PySequence_Fast(files_obj, "files must be a sequence of strings"), true);
    if (!seq)
        verror("files must be a sequence of strings");
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*seq); ++i) {
        const char *s = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(*seq, i));
        if (!s)
            verror("files must be a sequence of strings");
        args.files.emplace_back(s);
    }
    if (args.files.empty())
        verror("Files argument is not a vector of strings");
}

} // namespace

// pm_track_2d_import_files(track_dir, files, max_data_size, max_mem_usage_kb) -> dict
//
// gtrack_2d_import: reads the files (header chrom1 .. end2, then the value) and writes the pair files of the
// track into track_dir, which exists. A POINTS track if every interval is 1 x 1, a RECTS track otherwise.
// Intervals on a chromosome the database does not have are skipped. Returns the import_chroms_result; nothing
// is written when no chromosome pair got data.
PyObject *pm_track_2d_import_files(PyObject *self, PyObject *args)
{
    (void)self;
    try {
        const char *dir_str;
        PyObject *files_obj;
        long long max_data_size, max_mem_usage_kb;
        ImportArgs import_args;
        vector<PairFiles> pairs;
        UnknownChroms unknown;
        uint64_t num_matched;
        bool are_all_points = true;

        if (!PyArg_ParseTuple(args, "sOLL", &dir_str, &files_obj, &max_data_size, &max_mem_usage_kb))
            return nullptr;

        // STAGE 1: read the input files and split them into binary files each holding the intervals of a
        // chromosome pair. Each kid reads a contiguous block of the files.
        {
            PyMisha pymisha(true);

            read_import_args(import_args, dir_str, files_obj, max_data_size, max_mem_usage_kb);

            vector<int64_t> sizes = get_files_sizes(import_args.files);
            vector<int> bounds = split_files(sizes, import_num_kids(sizes.size()));
            vector<char> kids_all_points = run_kids(bounds.size() - 1, [&](int kid) {
                    return (char)read_input_files(import_args, bounds[kid], bounds[kid + 1], kid);
                });

            for (char kid_all_points : kids_all_points)
                are_all_points &= (bool)kid_all_points;

            set<int> matched;
            load_kid_chroms(import_args.dirname, bounds.size() - 1, matched, unknown);
            num_matched = matched.size();
            pairs = list_pair_files(import_args.dirname);
            check_interrupt();
        }

        if (!pairs.empty())
            split_and_build(import_args, pairs, INTERVAL_RECORD_SIZE,
                [&](const PairFiles &pair_files) { return pair_num_subtrees(import_args, pair_files.size / INTERVAL_RECORD_SIZE); },
                [&](const PairFiles &pair_files) { return split_pair(import_args, pair_files); },
                are_all_points ? POINT_MEM : RECT_MEM,
                [&](const PairFiles &pair_files) { write_pair(import_args, pair_files, are_all_points); });

        return import_chroms_result(unknown, num_matched, pairs);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}

// pm_track_2d_import_contacts_files(track_dir, contacts, fends, allow_duplicates, max_data_size, max_mem_usage_kb) -> dict
//
// gtrack_import_contacts: reads the contacts files ("intervals-value" format if fends is None, "fends-value"
// otherwise) and writes the pair files of a POINTS track into track_dir, which exists. Contacts on a chromosome
// the database does not have are skipped. Returns the import_chroms_result, the chromosome names those of the
// fends file if there is one; nothing is written when no chromosome pair got data.
PyObject *pm_track_2d_import_contacts_files(PyObject *self, PyObject *args)
{
    (void)self;
    try {
        const char *dir_str;
        PyObject *files_obj, *fends_obj;
        int allow_duplicates;
        long long max_data_size, max_mem_usage_kb;
        ImportArgs import_args;
        vector<PairFiles> pairs;
        UnknownChroms unknown;
        uint64_t num_matched;

        if (!PyArg_ParseTuple(args, "sOOpLL", &dir_str, &files_obj, &fends_obj, &allow_duplicates, &max_data_size, &max_mem_usage_kb))
            return nullptr;

        string fends;
        if (fends_obj != Py_None) {
            const char *s = PyUnicode_AsUTF8(fends_obj);
            if (!s)
                return nullptr;
            fends = s;
        }

        // STAGE 1: read the contacts and split them into binary files each holding the contacts of a pair of
        // chromosomes
        {
            PyMisha pymisha(true);

            read_import_args(import_args, dir_str, files_obj, max_data_size, max_mem_usage_kb);
            if (fends_obj == Py_None)
                num_matched = process_contacts_as_intervals(import_args, unknown);
            else
                num_matched = process_contacts_as_fends(import_args, fends.c_str(), unknown);

            pairs = list_pair_files(import_args.dirname);
            check_interrupt();
        }

        if (!pairs.empty())
            split_and_build(import_args, pairs, CONTACT_RECORD_SIZE,
                [&](const PairFiles &pair_files) { return contacts_num_subtrees(import_args, pair_files); },
                [&](const PairFiles &pair_files) { return split_contacts_pair(import_args, pair_files); },
                CONTACT_MEM,
                [&](const PairFiles &pair_files) { write_contacts_pair(import_args, pair_files, allow_duplicates); });

        return import_chroms_result(unknown, num_matched, pairs);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}
