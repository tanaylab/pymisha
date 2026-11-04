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
// memory throttle of the stage 1 and 2 kids are not ported.

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
#include <functional>
#include <map>
#include <numeric>
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

    PyMisha::prepare4multitasking();

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

// Reads the next interval of an input file. Returns 0 at EOF, -1 for a line on a chromosome the database does
// not have (skipped, as R skips it), 1 otherwise.
int read_interval_line(const ImportArgs &args, BufferedFile &infile, vector<string> &fields, long &lineno,
                       int &chromid1, int &chromid2, int64_t &start1, int64_t &end1, int64_t &start2, int64_t &end2, float &val)
{
    char *endptr;

    lineno += split_line(infile, fields, '\t');
    check_interrupt();

    if (fields.empty())
        return 0;

    if (fields.size() < NUM_2D_COLS + 1)
        verror("File %s, line %ld: invalid format", infile.file_name().c_str(), lineno);

    try {
        chromid1 = args.chromkey->chrom2id(fields[CHROM1]);
        chromid2 = args.chromkey->chrom2id(fields[CHROM2]);
    } catch (TGLException &) {
        // there might be unrecognized chromosomes, ignore them
        return -1;
    }

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

        while ((res = read_interval_line(args, infile, fields, lineno, chromid1, chromid2, start1, end1, start2, end2, val))) {
            if (res < 0)
                continue;

            are_all_points &= start1 == end1 - 1 && start2 == end2 - 1;
            write_interval(out_files.get(args.dirname, chromid1, chromid2, kid), start1, end1, start2, end2, val);
        }
    }

    out_files.close();
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

typedef unordered_map<pair<uint64_t, uint64_t>, double> Contacts;

const int64_t CONTACT_RECORD_SIZE = 2 * sizeof(int64_t) + sizeof(float);

// Peak memory of building a chromosome pair, in bytes per contact (see pair_mem): the hash map that sums the
// duplicates plus the quad trees, which hold every contact twice (mirrored in a cis pair, one tree per
// orientation in a trans pair). Measured by R misha as the peak RSS of a serial import of one pair of 2M - 40M
// Hi-C-like contacts: at most 202.7 bytes a contact, cis and trans alike. Rounded up to a multiple of 8.
const uint64_t CONTACT_MEM = 208;

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

// STAGE 1, intervals-value format: each kid reads a contiguous block of the files
void process_contacts_as_intervals(const ImportArgs &args)
{
    vector<int64_t> sizes = get_files_sizes(args.files);
    vector<int> bounds = split_files(sizes, import_num_kids(sizes.size()));

    run_kids(bounds.size() - 1, [&](int kid) {
        PairOutFiles out_files;
        int64_t start1, start2, end1, end2;
        float val;

        for (int ifile = bounds[kid]; ifile < bounds[kid + 1]; ++ifile) {
            vector<string> fields;
            long lineno = 0;
            BufferedFile infile;
            int chromid1, chromid2, res;

            if (!open_intervals_file(infile, args.files[ifile], fields, lineno))
                continue;

            while ((res = read_interval_line(args, infile, fields, lineno, chromid1, chromid2, start1, end1, start2, end2, val))) {
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
        return (char)0;
    });
}

// STAGE 1, fends-value format: the fends file is read here, then each kid reads a contiguous block of the
// contacts files
void process_contacts_as_fends(const ImportArgs &args, const char *fends_fname)
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

    vector<int> chromids;
    vector<int64_t> coords;
    vector<bool> defined;
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

        fendid = strtoll(fendid_str, &endptr, 10);
        if (*endptr)
            verror("Error reading fragment ends file %s, line %ld: invalid %s", fends_fname, (long)lineno, FENDS_COLS[COL_FEND]);

        if (fendid < 0)
            verror("Error reading fragment ends file %s, line %ld: %s is out of range", fends_fname, (long)lineno, FENDS_COLS[COL_FEND]);

        // Grow only: ids need not be sorted, and shrinking drops earlier ones.
        if ((uint64_t)fendid >= chromids.size()) {
            chromids.resize(fendid + 1);
            coords.resize(fendid + 1);
            defined.resize(fendid + 1);
        }
        defined[fendid] = true;

        // Use chromosome name as-is (no forced chr prefix)
        chromids[fendid] = args.chromkey->chrom2id(chr_str);

        coord = strtoll(coord_str, &endptr, 10);
        if (*endptr)
            verror("Error reading fragment ends file %s, line %ld: invalid %s", fends_fname, (long)lineno, FENDS_COLS[COL_COORD]);

        if (coord < 0 || coord >= (int64_t)args.chromkey->get_chrom_size(chromids[fendid]))
            verror("Error reading fragment ends file %s, line %ld: %s is out of range", fends_fname, (long)lineno, FENDS_COLS[COL_COORD]);

        coords[fendid] = coord;
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
                if (fend1 < 0 || fend1 >= (int64_t)defined.size() || !defined[fend1])
                    continue;

                fend2 = strtoll(fend2_str, &endptr, 10);
                if (*endptr)
                    verror("Error reading contacts file %s, line %ld: invalid %s", contacts_fname, (long)lineno, CONTACTS_COLS[COL_FEND2]);
                if (fend2 < 0 || fend2 >= (int64_t)defined.size() || !defined[fend2])
                    continue;

                value = strtod(value_str, &endptr);
                if (*endptr)
                    verror("Error reading contacts file %s, line %ld: invalid %s", contacts_fname, (long)lineno, CONTACTS_COLS[COL_COUNT]);

                if (chromids[fend1] > chromids[fend2] || (chromids[fend1] == chromids[fend2] && coords[fend1] > coords[fend2]))
                    swap(fend1, fend2);

                write_contact(out_files.get(args.dirname, chromids[fend1], chromids[fend2], kid), coords[fend1], coords[fend2], value);
                check_interrupt();
            }
        }
        out_files.close();
        return (char)0;
    });
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

    // Read the contacts of a subtree, sum up the duplicates and insert them to StatQuadTreeCachedSerializer.
    for (const vector<string> &srcs : subtree_sources(args, pair_files, num_subtrees)) {
        Contacts contacts;

        for_each_contact(srcs, [&](int64_t coord1, int64_t coord2, float value) {
            pair<uint64_t, uint64_t> key(coord1, coord2);
            Contacts::iterator icontact = contacts.find(key);

            if (icontact == contacts.end())
                contacts[key] = value;
            else {
                if (!allow_duplicates)
                    verror("Duplicated contact (%s, %ld)-(%s, %ld)", args.chromkey->id2chrom(chromid1).c_str(), (long)coord1,
                           args.chromkey->id2chrom(chromid2).c_str(), (long)coord2);
                icontact->second += value;
            }
        });

        // Insert in coordinate order, not hash order: the hash order depends on the
        // C++ runtime library, and so did the file's bytes.
        vector<pair<pair<uint64_t, uint64_t>, double>> sorted_contacts(contacts.begin(), contacts.end());
        sort(sorted_contacts.begin(), sorted_contacts.end());

        for (auto icontact = sorted_contacts.cbegin(); icontact != sorted_contacts.cend(); ++icontact) {
            int64_t coord1 = icontact->first.first;
            int64_t coord2 = icontact->first.second;
            float value = icontact->second;

            qtree_serializer1.insert(PointsQuadTree::ValueType(coord1, coord2, value));

            if (chromid1 != chromid2)
                qtree_serializer2.insert(PointsQuadTree::ValueType(coord2, coord1, value));
            else if (num_subtrees == 1 && coord1 != coord2)
                qtree_serializer1.insert(PointsQuadTree::ValueType(coord2, coord1, value));

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

// pm_track_2d_import_files(track_dir, files, max_data_size, max_mem_usage_kb)
//
// gtrack_2d_import: reads the files (header chrom1 .. end2, then the value) and writes the pair files of the
// track into track_dir, which exists. A POINTS track if every interval is 1 x 1, a RECTS track otherwise.
PyObject *pm_track_2d_import_files(PyObject *self, PyObject *args)
{
    (void)self;
    try {
        const char *dir_str;
        PyObject *files_obj;
        long long max_data_size, max_mem_usage_kb;
        ImportArgs import_args;
        vector<PairFiles> pairs;
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

            pairs = list_pair_files(import_args.dirname);
            check_interrupt();
        }

        split_and_build(import_args, pairs, INTERVAL_RECORD_SIZE,
            [&](const PairFiles &pair_files) { return pair_num_subtrees(import_args, pair_files.size / INTERVAL_RECORD_SIZE); },
            [&](const PairFiles &pair_files) { return split_pair(import_args, pair_files); },
            are_all_points ? POINT_MEM : RECT_MEM,
            [&](const PairFiles &pair_files) { write_pair(import_args, pair_files, are_all_points); });

        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}

// pm_track_2d_import_contacts_files(track_dir, contacts, fends, allow_duplicates, max_data_size, max_mem_usage_kb)
//
// gtrack_import_contacts: reads the contacts files ("intervals-value" format if fends is None, "fends-value"
// otherwise) and writes the pair files of a POINTS track into track_dir, which exists.
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
                process_contacts_as_intervals(import_args);
            else
                process_contacts_as_fends(import_args, fends.c_str());

            pairs = list_pair_files(import_args.dirname);
            check_interrupt();
        }

        split_and_build(import_args, pairs, CONTACT_RECORD_SIZE,
            [&](const PairFiles &pair_files) { return contacts_num_subtrees(import_args, pair_files); },
            [&](const PairFiles &pair_files) { return split_contacts_pair(import_args, pair_files); },
            CONTACT_MEM,
            [&](const PairFiles &pair_files) { write_contacts_pair(import_args, pair_files, allow_duplicates); });

        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}
