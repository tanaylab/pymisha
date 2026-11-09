/*
 * pmutils.h
 *
 * Python misha utility functions (R-independent replacements for rdbutils)
 */

#ifndef PMUTILS_H_
#define PMUTILS_H_

#include <cstdint>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <cstring>
// drand48() is POSIX and declared here. This used to arrive transitively, which
// held until conda-forge's macOS toolchain moved (2026-09-03) and every
// arm64-apple-darwin conda build failed with "use of undeclared identifier
// 'drand48'". The Apple-clang wheels and the Linux builds were unaffected, so
// the include was missing all along rather than newly needed.
#include <stdlib.h>

#include "TGLException.h"

namespace pm {

// Get chromosome files from a track directory
// Filters out hidden files, directories, and known non-data files
inline void get_chrom_files(const char *dirname, std::vector<std::string> &chrom_files) {
    chrom_files.clear();

    DIR *dir = opendir(dirname);
    if (!dir) {
        TGLError("Cannot open directory %s: %s", dirname, strerror(errno));
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        const char *name = entry->d_name;

        // Skip hidden files and special entries
        if (name[0] == '.')
            continue;

        // Skip known non-data files
        if (strcmp(name, ".attributes") == 0 ||
            strcmp(name, "track.idx") == 0 ||
            strcmp(name, "track.dat") == 0 ||
            strcmp(name, "meta.yaml") == 0) {
            continue;
        }

        // Check if it's a regular file
        std::string fullpath = std::string(dirname) + "/" + name;
        struct stat st;
        if (stat(fullpath.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            chrom_files.push_back(name);
        }
    }

    closedir(dir);
}

// Random draws follow numpy's global generator, as R misha's follow .Random.seed: the first draw
// of a call seeds drand48 from a draw of numpy.random (rng_load), so numpy.random.seed() reproduces
// the call and the next call draws other numbers. A forked kid, or a worker of a Python pool, mixes
// its index into that seed, so that the kids draw different numbers.
extern bool g_rng_loaded;    // drand48 holds this call's state (reset by each top-level call)
extern int  g_rng_kid_idx;   // the index of this kid or pool worker, -1 in the main process
// The top-level calls continue one random stream (pm_rng_stream): a gextract(file=) that runs one
// call per part of its scope draws what one call over the whole scope draws
extern bool g_rng_stream;

// Seeds drand48 from numpy's global generator (and the kid index); defined in pymisha.cpp
void rng_load();

// A uniform draw in [0, 1)
inline double pm_rnd_func() {
    if (!g_rng_loaded)
        rng_load();
    return drand48();
}

// Seeds drand48 for this call with an explicit seed (a function's own seed argument)
inline void rng_seed(long seed) {
    srand48(seed);
    g_rng_loaded = true;
}

// Mixes a kid's index into a seed drawn from the parent's state
inline uint32_t kid_seed(uint32_t seed, int kid_idx) {
    return seed ^ (0x9E3779B9u * (uint32_t)(kid_idx + 1));
}

// In a kid forked by the C++ layer: when the parent has drawn already in this call, the kid seeds
// from a draw of the state it inherited mixed with its index; otherwise its first draw loads from
// numpy's generator (as inherited) mixed with its index.
inline void reseed_kid_rng(int kid_idx) {
    g_rng_kid_idx = kid_idx;
    if (g_rng_loaded)
        srand48((long)kid_seed((uint32_t)(drand48() * 4294967296.0), kid_idx));
}

} // namespace pm

#endif /* PMUTILS_H_ */
