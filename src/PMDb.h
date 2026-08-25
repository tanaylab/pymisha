/*
 * PMDb.h
 *
 * Database management for pymisha
 * Manages chromosome key, track listing, and database state
 */

#ifndef PMDB_H_
#define PMDB_H_

#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <unordered_map>

#include <Python.h>

#include "GenomeChromKey.h"

class PMDb {
public:
    PMDb();
    ~PMDb();

    // Database initialization
    void init(const std::string &groot, const std::string &uroot);
    void reload();
    void unload();

    // Cache invalidation (called by init/reload/unload internally; also
    // safe to call from anything that mutates the chromosome key).
    void invalidate_caches();
    static void clear_index_caches();

    // Build (and cache) the chrom/start/end DataFrame used by
    // pm_intervals_all. Returns a NEW reference: callers may either steal it
    // (e.g. PyTuple_SetItem) or Py_DECREF when done.
    PyObject *get_intervals_all_py() const;

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

    // Interval-set listing (cached alongside the track scan so listing
    // interval sets is O(1) instead of a Python rglob over tracks/).
    std::vector<std::string> interv_names() const;

    // Incrementally register / unregister an interval-set name without
    // rebuilding the whole cache.  Used by gintervals_save / _rm so a
    // freshly created (or removed) set becomes visible immediately
    // without paying the O(N tracks) pm_dbreload rescan.
    void register_interv(const std::string &name) const;
    void unregister_interv(const std::string &name) const;

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

    // Interval-set names found alongside tracks (suffixes .interv / .interv2d).
    // Same cache lifetime as m_track_cache: refreshed by rebuild_track_cache().
    mutable std::set<std::string> m_interv_cache;

    // pm_intervals_all cache.
    //
    // Rationale (E.1.2): on million-contig databases the current
    // pm_intervals_all rebuild dominates (millions of string copies +
    // pandas allocations). Since the chromosome key is fixed between
    // gdb_init/gdb_reload calls, we can serve repeated calls from cache.
    //
    // We cache the raw name/size vectors (not the PyObject) and rebuild a
    // fresh PMDataFrame on each call. This is intentional:
    //   1. Each caller gets independent numpy arrays so in-place mutation
    //      of the returned DataFrame can never corrupt the cache.
    //   2. Avoids any Py reference juggling across the cached PyObject's
    //      lifetime (which would be entangled with PMDb's lifetime even
    //      after Python shutdown).
    //   3. The cost is just N PyUnicode_FromString + 2 numpy allocations -
    //      orders of magnitude cheaper than ck.id2chrom() / id2size().
    mutable std::vector<std::string> m_intervals_all_names;
    mutable std::vector<int64_t> m_intervals_all_sizes;
    mutable bool m_intervals_all_built{false};

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
