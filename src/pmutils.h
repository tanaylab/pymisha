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

// Random function for sampling (using drand48)
inline double pm_rnd_func() {
    return drand48();
}

} // namespace pm

#endif /* PMUTILS_H_ */
