/*
 * PMDb.h
 *
 * Database management for pymisha
 * Manages chromosome key, track listing, and database state
 */

#ifndef PMDB_H_
#define PMDB_H_

#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <unordered_map>

#include "GenomeChromKey.h"

class PMDb {
public:
    PMDb();
    ~PMDb();

    // Database initialization
    void init(const std::string &groot, const std::string &uroot);
    void reload();
    void unload();

    // State accessors
    bool is_initialized() const { return m_initialized; }
    const std::string &groot() const { return m_groot; }
    const std::string &uroot() const { return m_uroot; }
    const std::vector<std::string> &datasets() const { return m_datasets; }

    // Chromosome key
    GenomeChromKey &chromkey() { return m_chromkey; }
    const GenomeChromKey &chromkey() const { return m_chromkey; }

    // Track operations
    std::vector<std::string> track_names() const;
    std::string track_path(const std::string &track_name) const;
    bool track_exists(const std::string &track_name) const;
    std::string track_dataset(const std::string &track_name) const;

    // Dataset management
    void set_datasets(const std::vector<std::string> &datasets);

    // Track attributes
    std::string track_attrs_path(const std::string &track_name) const;

private:
    bool m_initialized;
    std::string m_groot;      // Global database root
    std::string m_uroot;      // User database root
    std::vector<std::string> m_datasets;  // Additional datasets (in load order)

    GenomeChromKey m_chromkey;
    mutable std::set<std::string> m_track_cache;  // Cached track names
    mutable std::unordered_map<std::string, std::string> m_track_db; // Track -> db root

    // Load chromosome sizes from chrom_sizes.txt
    void load_chrom_sizes();

    // Scan for tracks in a database root
    void scan_tracks(const std::string &root, bool override) const;

    // Recursive track scanning helper
    void scan_tracks_impl(const std::string &base_dir,
                          const std::string &prefix,
                          const std::string &root,
                          bool override) const;

    // Check if a directory is a track directory
    bool is_track_dir(const std::string &path) const;

    // Rebuild track cache/map based on current roots/datasets
    void rebuild_track_cache();
};

// Global database instance (singleton-like pattern, matches pynaryn)
extern PMDb *g_pmdb;

#endif /* PMDB_H_ */
