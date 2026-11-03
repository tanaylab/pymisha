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
#include <climits>
#include <cstdlib>

#include "PMDb.h"
#include "PMDataFrame.h"
#include "TGLException.h"
#include "GenomeTrack.h"
#include "TrackIndex2D.h"
#include "GenomeIndex.h"

// Global database instance
PMDb *g_pmdb = nullptr;

PMDb::PMDb() : m_initialized(false) {}

PMDb::~PMDb() {}

void PMDb::init(const std::string &groot, const std::string &uroot) {
    // Verify groot exists
    struct stat st;
    if (stat(groot.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        TGLError("Database root directory does not exist: %s", groot.c_str());
    }

    // The chromosomes are read and checked before the loaded database is let go, as R's
    // gsetroot does: a database that fails leaves the loaded one loaded
    GenomeChromKey chromkey;
    std::string warning;
    load_chrom_sizes(groot, chromkey, warning);

    if (m_initialized) {
        unload();
    }

    invalidate_caches();
    clear_index_caches();

    m_groot = groot;
    m_uroot = uroot;
    m_datasets.clear();
    m_gwd = groot + "/tracks";
    m_chromkey = std::move(chromkey);
    {
        std::lock_guard<std::mutex> lock(m_chrom_files_mutex);
        m_chrom_files.clear();
    }
    m_init_warning = warning;

    // Cache track list
    rebuild_track_cache();

    m_initialized = true;
}

void PMDb::reload() {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    invalidate_caches();

    // Re-scan tracks
    rebuild_track_cache();
}

void PMDb::unload() {
    invalidate_caches();
    clear_index_caches();
    m_groot.clear();
    m_uroot.clear();
    m_datasets.clear();
    m_gwd.clear();
    m_gwd_root.clear();
    m_gwd_subdir = false;
    m_track_cache.clear();
    m_track_db.clear();
    m_interv_cache.clear();
    m_chromkey = GenomeChromKey();  // Reset
    m_init_warning.clear();
    {
        std::lock_guard<std::mutex> lock(m_chrom_files_mutex);
        m_chrom_files.clear();
    }
    m_initialized = false;
}

// Wholesale drop of the process-static per-directory index caches. Only for a
// database swap: a groot rebuilt or converted under the same absolute path would
// otherwise keep being read through the old layout, and unlike R misha there is no
// gdb.reload(rescan = True) to recover a poisoned session. misha's defect F.
void PMDb::clear_index_caches() {
    GenomeTrack::clear_index_cache();
    TrackIndex2D::clear_cache();
}

void PMDb::invalidate_caches() {
    // Deliberately does NOT touch the per-directory track-index caches: reload()
    // runs after every mutation, and dropping every parsed index there would
    // re-read them all on the next query. Those caches are cleared wholesale only
    // on a database swap (init/unload, below), and per directory at each site that
    // changes what lives at a path. Same split as misha, whose
    // gdb.reload(rescan = FALSE) also leaves the index cache alone.
    m_intervals_all_names.clear();
    m_intervals_all_names.shrink_to_fit();
    m_intervals_all_sizes.clear();
    m_intervals_all_sizes.shrink_to_fit();
    m_intervals_all_built = false;
}

PyObject *PMDb::get_intervals_all_py() const {
    // See PMDb.h for the cache-vectors-not-pyobject rationale.
    if (!m_intervals_all_built) {
        const GenomeChromKey &ck = m_chromkey;
        uint64_t n = ck.get_num_chroms();
        m_intervals_all_names.resize(n);
        m_intervals_all_sizes.resize(n);
        for (uint64_t i = 0; i < n; ++i) {
            m_intervals_all_names[i] = ck.id2chrom(i);
            m_intervals_all_sizes[i] = (int64_t)ck.get_chrom_size(i);
        }
        m_intervals_all_built = true;
    }

    uint64_t n = m_intervals_all_names.size();
    PMDataFrame df(n, 3, "intervals");
    df.init_col(0, "chrom", PMDataFrame::STR);
    df.init_col(1, "start", PMDataFrame::LONG);
    df.init_col(2, "end", PMDataFrame::LONG);
    for (uint64_t i = 0; i < n; ++i) {
        df.val_str(i, 0, m_intervals_all_names[i].c_str());
        df.val_long(i, 1, 0);
        df.val_long(i, 2, (long)m_intervals_all_sizes[i]);
    }
    PMPY result = df.construct_py(true);
    result.to_be_stolen();
    return (PyObject *)result;
}

extern unsigned g_transact_id;

const GenomeTrack::ChromFiles &PMDb::chrom_files(const std::string &dir) const {
    std::lock_guard<std::mutex> lock(m_chrom_files_mutex);
    if (m_chrom_files_transact != g_transact_id) {
        m_chrom_files.clear();
        m_chrom_files_transact = g_transact_id;
    }
    auto it = m_chrom_files.find(dir);
    if (it == m_chrom_files.end())
        it = m_chrom_files.emplace(dir, std::unique_ptr<GenomeTrack::ChromFiles>(new GenomeTrack::ChromFiles(m_chromkey, dir))).first;
    return *it->second;
}

void read_chrom_sizes(const std::string &groot, std::vector<std::string> &names,
                      std::vector<uint64_t> &sizes) {
    std::string chrom_file = groot + "/chrom_sizes.txt";

    std::ifstream ifs(chrom_file);
    if (!ifs) {
        TGLError("Cannot open chromosome sizes file: %s", chrom_file.c_str());
    }

    names.clear();
    sizes.clear();
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;

        std::istringstream iss(line);
        std::string chrom_name;
        uint64_t chrom_size;

        if (!(iss >> chrom_name >> chrom_size)) {
            TGLError("Invalid format in chromosome sizes file: %s", line.c_str());
        }
        names.push_back(chrom_name);
        sizes.push_back(chrom_size);
    }
}

static bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool has_chr_prefix(const std::string &name) {
    return name.compare(0, 3, "chr") == 0;
}

bool is_per_chromosome_db(const std::string &groot, const std::vector<std::string> &names) {
    // indexed as gdb_convert_to_indexed and the sequence readers see it: genome.idx and
    // genome.seq both, not a genome.idx left over from an interrupted conversion
    if (file_exists(groot + "/seq/genome.idx") && file_exists(groot + "/seq/genome.seq"))
        return false;
    if (names.empty())
        return false;

    std::vector<const std::string *> no_prefix;
    for (const auto &name : names) {
        if (!has_chr_prefix(name))
            no_prefix.push_back(&name);
    }
    if (no_prefix.empty())
        return false;
    if ((double)no_prefix.size() < 0.8 * (double)names.size())
        return false;

    // the first five unprefixed names: each has a chr<name>.seq and none a <name>.seq
    const size_t num_samples = std::min(no_prefix.size(), (size_t)5);
    for (size_t i = 0; i < num_samples; ++i) {
        if (!file_exists(groot + "/seq/chr" + *no_prefix[i] + ".seq"))
            return false;
    }
    for (size_t i = 0; i < num_samples; ++i) {
        if (file_exists(groot + "/seq/" + *no_prefix[i] + ".seq"))
            return false;
    }
    return true;
}

static std::string strip_chr(const std::string &name) {
    return has_chr_prefix(name) ? name.substr(3) : name;
}

static std::string real_path(const std::string &path) {
    char buf[PATH_MAX];
    return realpath(path.c_str(), buf) ? std::string(buf) : path;
}

static std::string dir_name(const std::string &path) {
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? "." : pos == 0 ? "/" : path.substr(0, pos);
}

std::string check_genome_idx(const std::string &groot, const std::vector<std::string> &names,
                             const std::vector<uint64_t> &sizes)
{
    const std::string seq_dir = groot + "/seq";
    if (!file_exists(seq_dir + "/genome.idx") || !file_exists(seq_dir + "/genome.seq"))
        return "";

    GenomeIndex index;
    index.load(seq_dir + "/genome.idx");
    const std::vector<ContigIndexEntry> &entries = index.get_all_entries();
    std::vector<std::string> idx_names(entries.size());
    std::vector<uint64_t> idx_lengths(entries.size());
    for (const ContigIndexEntry &entry : entries) {
        if (entry.chromid >= entries.size())
            TGLError("Chrom id %u in %s/genome.idx is out of range (%zu contigs)", entry.chromid,
                     seq_dir.c_str(), entries.size());
        idx_names[entry.chromid] = entry.name;
        idx_lengths[entry.chromid] = entry.length;
    }

    auto mismatch = [&](const std::string &what) {
        // a seq/ that is another database's: that database's chrom_sizes.txt is the one to match
        std::string owner = dir_name(real_path(seq_dir));
        std::string hint;
        if (owner != real_path(groot) && file_exists(owner + "/chrom_sizes.txt"))
            hint = " Its seq/ is in " + owner + ": if that database was converted after this one was made from it, copy " +
                   owner + "/chrom_sizes.txt into " + groot + ".";
        TGLError("%s/genome.idx does not match chrom_sizes.txt in %s: %s, so chromosomes would read other chromosomes' sequence.%s",
                 seq_dir.c_str(), groot.c_str(), what.c_str(), hint.c_str());
    };
    auto describe = [](const std::vector<std::string> &n, const std::vector<uint64_t> &l, size_t i) {
        return i < n.size() ? n[i] + " (" + std::to_string(l[i]) + " bp)" : std::string("missing");
    };

    const size_t n = sizes.size();
    for (size_t i = 0; i < std::max(n, idx_lengths.size()); ++i) {
        if (i >= n || i >= idx_lengths.size() || idx_lengths[i] != sizes[i]) {
            if (i >= n)  // chrom_sizes.txt may list only the index's first contigs
                break;
            mismatch("chrom id " + std::to_string(i) + " is " + describe(idx_names, idx_lengths, i) +
                     " in the index and " + describe(names, sizes, i) + " in chrom_sizes.txt");
        }
    }

    std::vector<size_t> differ;
    for (size_t i = 0; i < n; ++i) {
        if (idx_names[i] != names[i])
            differ.push_back(i);
    }
    if (differ.empty())
        return "";

    // A chromosome listed twice stops: by the same name, or by a name that differs from the
    // index's at its chrom id while its form with or without "chr" is listed at a chrom id the
    // index gives that name (as "chr1" and "1" for an index with only "chr1"). Names equal to the
    // index's (distinct contigs "chr1" and "1") are not compared so.
    std::unordered_map<std::string, std::pair<size_t, size_t>> positions;  // first, last
    for (size_t i = 0; i < n; ++i) {
        auto res = positions.emplace(names[i], std::make_pair(i, i));
        if (!res.second)
            res.first->second.second = i;
    }
    for (size_t d : differ) {
        const auto &pos = positions[names[d]];
        size_t twin = SIZE_MAX;
        if (pos.first != pos.second) {
            twin = pos.first != d ? pos.first : pos.second;
        } else {
            std::string other = has_chr_prefix(names[d]) ? names[d].substr(3) : "chr" + names[d];
            auto iother = positions.find(other);
            if (iother != positions.end() && strip_chr(idx_names[iother->second.first]) == strip_chr(names[d]))
                twin = iother->second.first;
        }
        if (twin != SIZE_MAX) {
            size_t lo = std::min(d, twin), hi = std::max(d, twin);
            TGLError("chrom_sizes.txt in %s lists the same chromosome twice: %s (chrom id %zu) and %s (chrom id %zu).",
                     groot.c_str(), names[lo].c_str(), lo, names[hi].c_str(), hi);
        }
    }

    // names that differ by more than the "chr" prefix
    std::vector<size_t> renamed;
    for (size_t i : differ) {
        if (strip_chr(idx_names[i]) != strip_chr(names[i]))
            renamed.push_back(i);
    }
    if (renamed.empty())
        return "";

    // a name of chrom_sizes.txt at another chrom id in the index
    std::unordered_map<std::string, size_t> idx_pos;
    for (size_t k = 0; k < renamed.size(); ++k)
        idx_pos.emplace(strip_chr(idx_names[renamed[k]]), k);
    for (size_t k = 0; k < renamed.size(); ++k) {
        auto imoved = idx_pos.find(strip_chr(names[renamed[k]]));
        if (imoved != idx_pos.end())
            mismatch(names[renamed[k]] + " is chrom id " + std::to_string(renamed[k]) + " in chrom_sizes.txt and " +
                     std::to_string(renamed[imoved->second]) + " in the index");
    }
    if (n < idx_lengths.size())
        mismatch("chrom_sizes.txt lists " + std::to_string(n) + " of the index's " + std::to_string(idx_lengths.size()) +
                 " contigs, which have to be its first ones, and chrom id " + std::to_string(renamed[0]) + " is " +
                 idx_names[renamed[0]] + " in the index and " + names[renamed[0]] + " in chrom_sizes.txt");

    return seq_dir + "/genome.idx names " + std::to_string(renamed.size()) + " of its " + std::to_string(n) +
           " contigs differently from chrom_sizes.txt in " + groot + " (chrom id " + std::to_string(renamed[0]) + " is " +
           idx_names[renamed[0]] + " in the index and " + names[renamed[0]] +
           " in chrom_sizes.txt). Sequence is read by chrom id and the sizes agree, so each chromosome reads the index "
           "contig at its chrom id; unless that is the same contig under another name, its sequence is wrong.";
}

namespace {

// The weights of the ASCII characters in ICU's root collation, which R's order() uses on a
// character vector when R is built with ICU and runs in a non-C locale; R misha 5.13.1 and
// later sorts chromosome names by this table whatever the locale. The primary order of
// the printable characters, as R's order() gives it:
//   space _ - , ; : ! ? . ' " ( ) [ ] { } @ * / \ & # % ` ^ + < = > | ~ $ 0-9 aA bB ... zZ
// The two cases of a letter share a primary weight and differ at the tertiary level,
// lowercase first. Every printable character has a primary weight (the root collation does
// not ignore punctuation), and none has a secondary weight of its own. Control characters
// are ignored.
struct AsciiCollation {
    int primary[128];
    int tertiary[128];

    AsciiCollation() {
        static const char order[] = " _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$0123456789";
        for (int c = 0; c < 128; ++c) {
            primary[c] = -1;
            tertiary[c] = 0;
        }
        int weight = 1;
        for (const char *p = order; *p; ++p)
            primary[(unsigned char)*p] = weight++;
        for (int c = 'a'; c <= 'z'; ++c) {
            primary[c] = weight;
            primary[c - 'a' + 'A'] = weight;
            tertiary[c - 'a' + 'A'] = 1;
            ++weight;
        }
    }
};

const AsciiCollation &ascii_collation() {
    static const AsciiCollation collation;
    return collation;
}

// The primary and tertiary weights of a name. A byte outside ASCII sorts after every ASCII
// character, by its value; ICU's order of non-ASCII characters is not reproduced.
void collation_key(const std::string &s, std::vector<int> &primary, std::vector<int> &tertiary) {
    const AsciiCollation &collation = ascii_collation();
    primary.clear();
    tertiary.clear();
    for (unsigned char c : s) {
        if (c < 128) {
            if (collation.primary[c] < 0)
                continue;
            primary.push_back(collation.primary[c]);
            tertiary.push_back(collation.tertiary[c]);
        } else {
            primary.push_back(1000 + c);
            tertiary.push_back(0);
        }
    }
}

}  // namespace

bool r_collate_less(const std::string &a, const std::string &b) {
    std::vector<int> pa, ta, pb, tb;
    collation_key(a, pa, ta);
    collation_key(b, pb, tb);
    if (pa != pb)
        return pa < pb;
    return ta < tb;
}

GdbChromOrder gdb_chrom_order(const std::string &groot, const std::vector<std::string> &chrom_sizes_names) {
    GdbChromOrder res;
    res.names = chrom_sizes_names;
    res.id_order.resize(res.names.size());
    for (size_t i = 0; i < res.id_order.size(); ++i)
        res.id_order[i] = i;
    res.per_chromosome = is_per_chromosome_db(groot, chrom_sizes_names);
    if (!res.per_chromosome)
        return res;

    for (auto &name : res.names) {
        if (!has_chr_prefix(name))
            name = "chr" + name;
    }
    // order(names); stable, as order() keeps ties in their original order
    std::vector<std::pair<std::vector<int>, std::vector<int>>> keys(res.names.size());
    for (size_t i = 0; i < res.names.size(); ++i)
        collation_key(res.names[i], keys[i].first, keys[i].second);
    std::stable_sort(res.id_order.begin(), res.id_order.end(),
                     [&keys](size_t i, size_t j) { return keys[i] < keys[j]; });
    return res;
}

void PMDb::load_chrom_sizes(const std::string &groot, GenomeChromKey &chromkey, std::string &warning) {
    std::vector<std::string> file_names;
    std::vector<uint64_t> file_sizes;
    read_chrom_sizes(groot, file_names, file_sizes);
    if (file_names.empty())
        TGLError("chrom_sizes.txt file does not contain any chromosomes");

    // the canonical names and the chrom ids that R misha's gsetroot gives (.gdb.chrom_order),
    // and an indexed seq/ checked against them
    GdbChromOrder order = gdb_chrom_order(groot, file_names);
    std::vector<std::string> names_by_id;
    std::vector<uint64_t> sizes_by_id;
    for (size_t idx : order.id_order) {
        names_by_id.push_back(order.names[idx]);
        sizes_by_id.push_back(file_sizes[idx]);
    }
    warning = check_genome_idx(groot, names_by_id, sizes_by_id);
    {
        std::unordered_set<std::string> seen;
        for (const auto &name : names_by_id) {
            if (!seen.insert(name).second)
                TGLError("Chromosome \"%s\" appears more than once in chrom_sizes.txt", name.c_str());
        }
    }
    chromkey = GenomeChromKey();
    for (size_t idx : order.id_order)
        chromkey.add_chrom(order.names[idx], file_sizes[idx]);

    // the aliases, added in chrom_sizes.txt order as R adds them (the first one wins)
    const std::vector<std::string> &chrom_names = order.names;

    // Pre-scan for the chr-prefix-alias skip heuristic (R misha #112):
    // On fragmented assemblies (e.g. 2.4M-contig scaffold_* genomes) the
    // "chr"+name aliases blow up the alias map for no user benefit, since
    // nobody types `chrscaffold_12345`. Skip them when the genome has more
    // than 1000 contigs AND no canonical name carries a "chr" prefix AND
    // no name is a mitochondrial pattern. Small genomes and any Ensembl/UCSC
    // style assembly with mito stay unaffected.
    bool has_chr_prefix_any = false;
    bool has_mito_any = false;
    for (const auto &chrom_name : chrom_names) {
        if (chrom_name.compare(0, 3, "chr") == 0) {
            has_chr_prefix_any = true;
        }
        std::string unpref = chrom_name;
        if (chrom_name.compare(0, 3, "chr") == 0 && chrom_name.size() > 3) {
            unpref = chrom_name.substr(3);
        }
        std::string up_unpref = unpref;
        std::transform(up_unpref.begin(), up_unpref.end(), up_unpref.begin(), ::toupper);
        std::string up_chrom = chrom_name;
        std::transform(up_chrom.begin(), up_chrom.end(), up_chrom.begin(), ::toupper);
        if (up_unpref == "M" || up_unpref == "MT" ||
            up_chrom == "CHRM" || up_chrom == "CHRMT") {
            has_mito_any = true;
        }
        if (has_chr_prefix_any && has_mito_any)
            break;
    }
    const bool add_chr_prefix_aliases =
        has_chr_prefix_any || has_mito_any || chrom_names.size() <= 1000;

    // Add aliases (chr prefix + mitochondrial aliases)
    for (const auto &chrom_name : chrom_names) {
        int id = chromkey.chrom2id(chrom_name);
        if (id < 0)
            continue;

        std::string unprefixed = chrom_name;
        if (chrom_name.compare(0, 3, "chr") == 0) {
            if (chrom_name.size() > 3) {
                unprefixed = chrom_name.substr(3);
                chromkey.add_chrom_alias(unprefixed, id);
            }
        } else if (add_chr_prefix_aliases) {
            chromkey.add_chrom_alias("chr" + chrom_name, id);
        }

        if (!has_mito_any)
            continue;

        std::string upper_unprefixed = unprefixed;
        std::transform(upper_unprefixed.begin(), upper_unprefixed.end(),
                       upper_unprefixed.begin(), ::toupper);
        std::string upper_chrom = chrom_name;
        std::transform(upper_chrom.begin(), upper_chrom.end(),
                       upper_chrom.begin(), ::toupper);
        if (upper_unprefixed == "M" || upper_unprefixed == "MT" ||
            upper_chrom == "CHRM" || upper_chrom == "CHRMT") {
            chromkey.add_chrom_alias("M", id);
            chromkey.add_chrom_alias("MT", id);
            chromkey.add_chrom_alias("chrM", id);
        }
    }
}

std::vector<std::string> PMDb::track_names() const {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    return std::vector<std::string>(m_track_cache.begin(), m_track_cache.end());
}

std::vector<std::string> PMDb::interv_names() const {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }

    return std::vector<std::string>(m_interv_cache.begin(), m_interv_cache.end());
}

void PMDb::register_interv(const std::string &name) const {
    m_interv_cache.insert(name);
}

void PMDb::unregister_interv(const std::string &name) const {
    m_interv_cache.erase(name);
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

    // in a subdirectory (gdir_cd) the names are relative to it
    if (m_gwd_subdir) {
        return m_gwd + "/" + path_name + ".track";
    }
    const std::string &root = it->second;
    return root + "/tracks/" + path_name + ".track";
}

std::string PMDb::new_track_path(const std::string &track_name) const {
    if (!m_initialized) {
        TGLError("Database not initialized. Call gdb_init() first.");
    }
    std::string path_name = track_name;
    std::replace(path_name.begin(), path_name.end(), '.', '/');
    const std::string base = !m_uroot.empty() && m_gwd_root == m_groot ? m_uroot + "/tracks" : m_gwd;
    return base + "/" + path_name + ".track";
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
        if (stat(full_path.c_str(), &st) != 0) {
            continue;
        }

        std::string name_str(name);

        // Detect interval-set names (.interv2d / .interv).  These can be a
        // single binary file *or* a per-chromosome directory (legacy
        // layout, e.g. .../rmsk.interv/chr1, chr2, ...).
        // Check .interv2d before .interv since .interv is a prefix of it.
        const std::string suffix2d = ".interv2d";
        const std::string suffix1d = ".interv";
        std::string interv_base;
        if (name_str.size() > suffix2d.size() &&
            name_str.compare(name_str.size() - suffix2d.size(), suffix2d.size(), suffix2d) == 0) {
            interv_base = name_str.substr(0, name_str.size() - suffix2d.size());
        } else if (name_str.size() > suffix1d.size() &&
                   name_str.compare(name_str.size() - suffix1d.size(), suffix1d.size(), suffix1d) == 0) {
            interv_base = name_str.substr(0, name_str.size() - suffix1d.size());
        }
        if (!interv_base.empty() && (S_ISREG(st.st_mode) || S_ISDIR(st.st_mode))) {
            std::string interv_name = prefix.empty() ? interv_base : (prefix + "." + interv_base);
            m_interv_cache.insert(interv_name);
            // Don't recurse into per-chrom interv dirs.
            continue;
        }

        if (!S_ISDIR(st.st_mode)) {
            continue;
        }

        // Check if this is a .track directory
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
    m_interv_cache.clear();

    // The root whose tracks/ the working directory is (R's gdb.reload compares GWD with
    // file.path(c(GROOT, GDATASETS), "tracks"))
    m_gwd_root.clear();
    if (!m_gwd.empty()) {
        const std::string gwd = real_path(m_gwd);
        std::vector<std::string> roots{m_groot};
        if (!m_uroot.empty()) {
            roots.push_back(m_uroot);
        }
        roots.insert(roots.end(), m_datasets.begin(), m_datasets.end());
        for (const auto &root : roots) {
            if (!root.empty() && real_path(root + "/tracks") == gwd) {
                m_gwd_root = root;
                break;
            }
        }
    }
    m_gwd_subdir = !m_gwd.empty() && m_gwd_root.empty();

    // In a subdirectory only the names under it are listed, relative to it
    if (m_gwd_subdir) {
        scan_tracks_impl(m_gwd, "", "", true);
        return;
    }

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
