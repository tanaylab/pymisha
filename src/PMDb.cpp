/*
 * PMDb.cpp
 *
 * Database management for pymisha
 */

#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstring>
#include <algorithm>
#include <memory>
#include <unordered_set>

#include "PMDb.h"
#include "TGLException.h"

// Global database instance
PMDb *g_pmdb = nullptr;

PMDb::PMDb() : m_initialized(false) {}

PMDb::~PMDb() {}

void PMDb::init(const std::string &groot, const std::string &uroot) {
    if (m_initialized) {
        unload();
    }

    m_groot = groot;
    m_uroot = uroot;
    m_datasets.clear();

    // Verify groot exists
    struct stat st;
    if (stat(groot.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        TGLError("Database root directory does not exist: %s", groot.c_str());
    }

    // Load chromosome sizes
    load_chrom_sizes();

    // Cache track list
    rebuild_track_cache();

    m_initialized = true;
}

void PMDb::reload() {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    // Re-scan tracks
    rebuild_track_cache();
}

void PMDb::unload() {
    m_groot.clear();
    m_uroot.clear();
    m_datasets.clear();
    m_track_cache.clear();
    m_track_db.clear();
    m_chromkey = GenomeChromKey();  // Reset
    m_initialized = false;
}

void PMDb::load_chrom_sizes() {
    std::string chrom_file = m_groot + "/chrom_sizes.txt";

    std::ifstream ifs(chrom_file);
    if (!ifs) {
        TGLError("Cannot open chromosome sizes file: %s", chrom_file.c_str());
    }

    std::string line;
    std::vector<std::string> chrom_names;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string chrom_name;
        uint64_t chrom_size;

        if (!(iss >> chrom_name >> chrom_size)) {
            TGLError("Invalid format in chromosome sizes file: %s", line.c_str());
        }

        // Add chromosome to key
        m_chromkey.add_chrom(chrom_name, chrom_size);
        chrom_names.push_back(chrom_name);
    }

    // Add aliases (chr prefix + mitochondrial aliases)
    for (const auto &chrom_name : chrom_names) {
        int id = m_chromkey.chrom2id(chrom_name);
        if (id < 0)
            continue;

        std::string unprefixed = chrom_name;
        if (chrom_name.compare(0, 3, "chr") == 0) {
            if (chrom_name.size() > 3) {
                unprefixed = chrom_name.substr(3);
                m_chromkey.add_chrom_alias(unprefixed, id);
            }
        } else {
            m_chromkey.add_chrom_alias("chr" + chrom_name, id);
        }

        std::string upper_unprefixed = unprefixed;
        std::transform(upper_unprefixed.begin(), upper_unprefixed.end(),
                       upper_unprefixed.begin(), ::toupper);
        std::string upper_chrom = chrom_name;
        std::transform(upper_chrom.begin(), upper_chrom.end(),
                       upper_chrom.begin(), ::toupper);
        if (upper_unprefixed == "M" || upper_unprefixed == "MT" ||
            upper_chrom == "CHRM" || upper_chrom == "CHRMT") {
            m_chromkey.add_chrom_alias("M", id);
            m_chromkey.add_chrom_alias("MT", id);
            m_chromkey.add_chrom_alias("chrM", id);
        }
    }
}

std::vector<std::string> PMDb::track_names() const {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    return std::vector<std::string>(m_track_cache.begin(), m_track_cache.end());
}

std::string PMDb::track_path(const std::string &track_name) const {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    // Replace dots with path separators for hierarchical track names
    std::string path_name = track_name;
    std::replace(path_name.begin(), path_name.end(), '.', '/');

    auto it = m_track_db.find(track_name);
    if (it == m_track_db.end()) {
        TGLError("Track not found: %s", track_name.c_str());
    }

    const std::string &root = it->second;
    return root + "/tracks/" + path_name + ".track";
}

bool PMDb::track_exists(const std::string &track_name) const {
    return m_track_db.find(track_name) != m_track_db.end();
}

std::string PMDb::track_dataset(const std::string &track_name) const {
    auto it = m_track_db.find(track_name);
    if (it == m_track_db.end()) {
        return "";
    }
    return it->second;
}

std::string PMDb::track_attrs_path(const std::string &track_name) const {
    return track_path(track_name) + "/.attributes";
}

void PMDb::scan_tracks(const std::string &root, bool override) const {
    std::string dir = root + "/tracks";
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        return;  // Directory doesn't exist, skip
    }

    scan_tracks_impl(dir, "", root, override);
}

void PMDb::scan_tracks_impl(const std::string &base_dir,
                            const std::string &prefix,
                            const std::string &root,
                            bool override) const {
    std::unique_ptr<DIR, int(*)(DIR *)> dir(opendir(base_dir.c_str()), closedir);
    if (!dir) return;

    struct dirent *entry;
    while ((entry = readdir(dir.get())) != nullptr) {
        const char *name = entry->d_name;

        // Skip hidden files and special entries
        if (name[0] == '.') continue;

        std::string full_path = base_dir + "/" + name;

        struct stat st;
        if (stat(full_path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            continue;  // Not a directory
        }

        // Check if this is a .track directory
        std::string name_str(name);
        if (name_str.size() > 6 && name_str.substr(name_str.size() - 6) == ".track") {
            // It's a track - extract name
            std::string track_name = name_str.substr(0, name_str.size() - 6);
            if (!prefix.empty()) {
                track_name = prefix + "." + track_name;
            }
            m_track_cache.insert(track_name);
            if (override || m_track_db.find(track_name) == m_track_db.end()) {
                m_track_db[track_name] = root;
            }
        } else if (name_str.find('.') == std::string::npos) {
            // It's a subdirectory (track set), recurse
            std::string new_prefix = prefix.empty() ? name_str : (prefix + "." + name_str);
            scan_tracks_impl(full_path, new_prefix, root, override);
        }
    }

}

bool PMDb::is_track_dir(const std::string &path) const {
    // A track directory has chromosome files inside it
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        return false;
    }

    // Check for any chromosome file
    for (uint64_t i = 0; i < m_chromkey.get_num_chroms(); ++i) {
        std::string chrom_file = path + "/" + m_chromkey.id2chrom(i);
        if (stat(chrom_file.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            return true;
        }
        // Also check chr prefix variant
        const std::string &chrom_name = m_chromkey.id2chrom(i);
        if (chrom_name.compare(0, 3, "chr") == 0) {
            chrom_file = path + "/" + chrom_name.substr(3);
        } else {
            chrom_file = path + "/chr" + chrom_name;
        }
        if (stat(chrom_file.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            return true;
        }
    }

    // Check for indexed format
    std::string idx_file = path + "/track.idx";
    return stat(idx_file.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

void PMDb::set_datasets(const std::vector<std::string> &datasets)
{
    m_datasets = datasets;
    if (m_initialized) {
        rebuild_track_cache();
    }
}

void PMDb::rebuild_track_cache()
{
    m_track_cache.clear();
    m_track_db.clear();

    // Datasets (load order); later datasets override earlier ones
    for (const auto &ds : m_datasets) {
        scan_tracks(ds, true);
    }

    // Working database overrides datasets
    if (!m_groot.empty()) {
        scan_tracks(m_groot, true);
    }

    // User database overrides all
    if (!m_uroot.empty()) {
        scan_tracks(m_uroot, true);
    }
}
