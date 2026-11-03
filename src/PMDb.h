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
#include <memory>
#include <mutex>
#include <unordered_map>

#include <Python.h>

#include "GenomeChromKey.h"
#include "GenomeTrack.h"

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

    // The warning the last init() gave (check_genome_idx), empty if none
    const std::string &init_warning() const { return m_init_warning; }

    // The per-chromosome files of a track or intervals set directory (GenomeTrack::ChromFiles),
    // listed once per call from Python, as R's IntervUtils::get_chrom_files lists them once per
    // call from R. Thread-safe.
    const GenomeTrack::ChromFiles &chrom_files(const std::string &dir) const;

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

    // The working directory (gdir_cd), as R misha's GWD. When it is a subdirectory, not the
    // tracks/ of the working database, the user root or a loaded dataset, the names listed are
    // those under it, relative to it (R's gdb.reload). Takes effect at the next reload().
    void set_gwd(const std::string &gwd) { m_gwd = gwd; }
    bool gwd_is_subdir() const { return m_gwd_subdir; }
    // The directory of a new track: the user root's tracks/ when one is set and the working
    // directory is the working database's tracks/, the working directory otherwise
    std::string new_track_path(const std::string &track_name) const;

    // Track attributes
    std::string track_attrs_path(const std::string &track_name) const;

private:
    bool m_initialized;
    std::string m_groot;      // Global database root
    std::string m_uroot;      // User database root
    std::vector<std::string> m_datasets;  // Additional datasets (in load order)
    std::string m_gwd;                    // Working directory (set_gwd)
    std::string m_gwd_root;               // The root whose tracks/ m_gwd is, empty if none
    bool m_gwd_subdir{false};             // m_gwd is not the tracks/ of any root

    GenomeChromKey m_chromkey;
    std::string m_init_warning;  // check_genome_idx's warning at init(), empty if none
    // chrom_files(), by directory, for the call g_transact_id names
    mutable std::unordered_map<std::string, std::unique_ptr<GenomeTrack::ChromFiles>> m_chrom_files;
    mutable unsigned m_chrom_files_transact{0};
    mutable std::mutex m_chrom_files_mutex;
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

    // The chromosomes of the database at groot, from its chrom_sizes.txt, as R misha's gsetroot
    // gives them: their key with the aliases, and the warning check_genome_idx gives (empty if
    // none). Throws where gsetroot stops.
    static void load_chrom_sizes(const std::string &groot, GenomeChromKey &chromkey, std::string &warning);

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

// The rows of <groot>/chrom_sizes.txt: chromosome names and sizes, in file order.
void read_chrom_sizes(const std::string &groot, std::vector<std::string> &names,
                      std::vector<uint64_t> &sizes);

// R misha's .is_per_chromosome_db (db-core.R): whether the database at groot, whose
// chrom_sizes.txt holds `names`, is a per-chromosome database whose chromosome names lack
// the "chr" prefix of its seq files: no seq/genome.idx, at least 80% of the names lack the
// prefix, and the first five of those have a seq/chr<name>.seq and no seq/<name>.seq.
bool is_per_chromosome_db(const std::string &groot, const std::vector<std::string> &names);

// R misha's .gdb.check_genome_idx (db-core.R): an indexed seq/ (genome.idx and genome.seq) of
// the database at groot is read by chrom id, so its contigs have to be the database's
// chromosomes (names, sizes: by chrom id) in that order. Throws where R stops: the sizes differ,
// a name is at another chrom id in the index, chrom_sizes.txt lists a chromosome twice (as chr1
// and 1), or it lists fewer contigs than the index with other names. Returns R's warning when
// only names differ, at the same chrom ids and with the same sizes (empty if none).
std::string check_genome_idx(const std::string &groot, const std::vector<std::string> &names,
                             const std::vector<uint64_t> &sizes);

// Whether name a sorts before name b in R's order() on a character vector, as R gives it
// when built with ICU and run in a non-C locale (ICU's root collation; R in a C locale
// orders by bytes instead). Exact for ASCII names. R misha 5.13.1 and later sorts the
// chromosomes of a per-chromosome database by the same fixed rule, whatever the locale.
bool r_collate_less(const std::string &a, const std::string &b);

// R misha's .gdb.chrom_order (db-core.R): the canonical chromosome names of the database at
// groot, whose chrom_sizes.txt holds chrom_sizes_names, and the order of their chrom ids, as
// gsetroot() sets them. A per-chromosome database (is_per_chromosome_db) gets the "chr"
// prefix of its seq files, and its chrom ids follow the sorted names (r_collate_less). Any
// other database keeps the chrom_sizes.txt names and order. An indexed track (track.idx) is
// keyed by these ids.
struct GdbChromOrder {
    std::vector<std::string> names;   // canonical names, in chrom_sizes.txt order
    std::vector<size_t> id_order;     // chrom id i is names[id_order[i]]
    bool per_chromosome;
};
GdbChromOrder gdb_chrom_order(const std::string &groot, const std::vector<std::string> &chrom_sizes_names);

// Global database instance (singleton-like pattern, matches pynaryn)
extern PMDb *g_pmdb;

#endif /* PMDB_H_ */
