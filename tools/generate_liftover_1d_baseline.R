# Regenerate tests/test_data/liftover_1d from R misha (5.12.1 or later).
#
# tests/test_liftover_1d_r_parity.py checks that gtrack_liftover of a sparse or a
# dense track gives the track R's gtrack.liftover gives: the intervals and values
# R reads back from each lifted track, and the size and SHA-256 of each file R
# wrote, are frozen in baseline.json, so the test needs no R. The source tracks
# are written by R too and kept in the fixture. Re-run only when the cases change:
#
#   Rscript tools/generate_liftover_1d_baseline.R tests/test_data/liftover_1d

suppressMessages(library(misha))
suppressMessages(library(jsonlite))
suppressMessages(library(digest))

out <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(out))
unlink(out, recursive = TRUE)
dir.create(out, recursive = TRUE)
out <- normalizePath(out)
work <- tempfile("liftover_1d_")
dir.create(work)

mkdb <- function(dir, chroms, sizes, format = "per-chromosome") {
    fa <- file.path(dir, "fa")
    dir.create(fa, recursive = TRUE)
    seqs <- sprintf(">%s\n%s\n", chroms, vapply(sizes, function(s) strrep("A", s), ""))
    if (format == "indexed") {
        fas <- file.path(fa, "genome.fasta")
        cat(seqs, sep = "", file = fas)
    } else {
        fas <- file.path(fa, paste0(chroms, ".fasta"))
        for (i in seq_along(chroms)) cat(seqs[i], file = fas[i])
    }
    db <- file.path(dir, "db")
    suppressMessages(gdb.create(groot = db, fasta = fas, format = format))
    db
}

set.seed(31)
src_chroms <- c("chrS1", "chrS2")
src_sizes <- c(2000, 1500)
src <- mkdb(file.path(work, "src"), src_chroms, src_sizes)
gdb.init(src)

# values with NaN, and an Inf or two, which R reads as NaN
vals <- function(n) {
    v <- sample(1:40, n, replace = TRUE) / 4
    v[sample(n, round(n / 8))] <- NaN
    v[sample(n, 2)] <- c(Inf, -Inf)
    v
}

# sparse: disjoint intervals of 1 to 60bp with gaps of 0 to 30bp
sp <- NULL
for (i in seq_along(src_chroms)) {
    pos <- 0
    repeat {
        pos <- pos + sample(0:30, 1)
        len <- sample(1:60, 1)
        if (pos + len > src_sizes[i]) break
        sp <- rbind(sp, data.frame(chrom = src_chroms[i], start = pos, end = pos + len))
        pos <- pos + len
    }
}
gtrack.create_sparse("sp", "x", sp, vals(nrow(sp)))
# dense: bins of 20bp
dn <- giterator.intervals(NULL, gintervals.all(), iterator = 20)
gtrack.create_dense("dn", "x", dn, vals(nrow(dn)), 20, NaN)

fix <- file.path(out, "src")
dir.create(file.path(fix, "tracks"), recursive = TRUE)
file.copy(file.path(src, "chrom_sizes.txt"), fix)
for (tr in c("sp", "dn")) {
    tdir <- file.path(fix, "tracks", paste0(tr, ".track"))
    dir.create(tdir)
    for (f in list.files(file.path(src, "tracks", paste0(tr, ".track"))))
        file.copy(file.path(src, "tracks", paste0(tr, ".track"), f), tdir)
}

# Chain blocks, given as (src chrom, src start, src end, tgt chrom, tgt strand, tgt start, id).
# The targets of chains 1 to 3 overlap one another at different offsets; chain 4
# lands on the reverse strand; chain 5 maps chrS1 [0, 500) a second time (a source
# overlap); chain 6 is two blocks with gaps in both genomes.
cat(
    "chain 1000 chrS1 2000 + 0 500 chrT1 8000 + 0 500 1\n500\n\n",
    "chain 900 chrS1 2000 + 1000 1500 chrT1 8000 + 137 637 2\n500\n\n",
    "chain 800 chrS2 1500 + 0 300 chrT1 8000 + 400 700 3\n300\n\n",
    "chain 700 chrS2 1500 + 500 1000 chrT1 8000 - 0 500 4\n500\n\n",
    "chain 600 chrS1 2000 + 0 500 chrT1 8000 + 2000 2500 5\n500\n\n",
    "chain 500 chrS2 1500 + 1000 1500 chrT2 6000 + 100 590 6\n200\t30\t20\n270\n\n",
    sep = "", file = file.path(out, "test.chain")
)

# no chain lands on chrT3
tgt_dbs <- list()
for (fmt in c("per-chromosome", "indexed"))
    tgt_dbs[[fmt]] <- mkdb(file.path(work, paste0("tgt_", fmt)), c("chrT1", "chrT2", "chrT3"), c(8000, 6000, 3000), fmt)

aggs <- c("mean", "median", "sum", "min", "max", "count", "first", "last", "nth",
          "max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac")
cases <- list()
add <- function(track, policy, agg, nth = NA, min_n = NA, na_rm = TRUE, tgt = "per-chromosome") {
    name <- paste(track, policy, gsub(".", "_", agg, fixed = TRUE), sep = "_")
    if (!is.na(nth)) name <- paste0(name, nth)
    if (!is.na(min_n)) name <- paste0(name, "_min_n", min_n)
    if (!na_rm) name <- paste0(name, "_na_rm_false")
    if (tgt == "indexed") name <- paste0(name, "_indexed")
    cases[[length(cases) + 1]] <<- list(name = name, track = track, policy = policy, agg = agg, nth = nth,
                                         min_n = min_n, na_rm = na_rm, tgt = tgt)
}
for (tr in c("sp", "dn")) {
    for (policy in c("keep", "agg", "auto")) {
        for (agg in aggs) add(tr, policy, agg, nth = if (agg == "nth") 2 else NA)
        add(tr, policy, "nth", nth = 3)
        add(tr, policy, "mean", na_rm = FALSE)
        add(tr, policy, "mean", min_n = 2)
        add(tr, policy, "count", min_n = 1)
    }
    add(tr, "keep", "mean", tgt = "indexed")
    add(tr, "agg", "count", tgt = "indexed")
}

for (i in seq_along(cases)) {
    cs <- cases[[i]]
    gdb.init(tgt_dbs[[cs$tgt]])
    covered <- gintervals(c("chrT1", "chrT1", "chrT1", "chrT2", "chrT3"), c(0, 2000, 7500, 100, 0),
                          c(700, 2500, 8000, 590, 100))
    args <- list(cs$name, "x", file.path(src, "tracks", paste0(cs$track, ".track")), file.path(out, "test.chain"),
                 src_overlap_policy = "keep", tgt_overlap_policy = cs$policy, multi_target_agg = cs$agg,
                 na.rm = cs$na_rm)
    if (!is.na(cs$nth)) args$params <- cs$nth
    if (!is.na(cs$min_n)) args$min_n <- cs$min_n
    suppressMessages(do.call(gtrack.liftover, args))
    # a sparse track's intervals, or a dense track's bins where the chains land
    r <- if (cs$track == "sp") gextract(cs$name, gintervals.all(), iterator = cs$name) else gextract(cs$name, covered, iterator = 20)
    cases[[i]]$intervals <- list(chrom = as.character(r$chrom), start = r$start, end = r$end,
                                 value = ifelse(is.nan(r[[cs$name]]), "NaN", sprintf("%.9g", r[[cs$name]])))
    tdir <- file.path(tgt_dbs[[cs$tgt]], "tracks", paste0(cs$name, ".track"))
    files <- sort(list.files(tdir))
    cases[[i]]$files <- lapply(setNames(files, files), function(f) {
        list(size = file.size(file.path(tdir, f)), sha256 = digest(file = file.path(tdir, f), algo = "sha256"))
    })
}

writeLines(toJSON(list(misha = as.character(packageVersion("misha")), cases = cases), auto_unbox = TRUE, pretty = TRUE,
                  digits = NA, na = "null"), file.path(out, "baseline.json"))
unlink(work, recursive = TRUE)
