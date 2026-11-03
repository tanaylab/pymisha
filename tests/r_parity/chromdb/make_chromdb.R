# Builds the fixture of tests/r_parity/test_chrom_names.py with R misha:
#   db/        a per-chromosome database whose chrom_sizes.txt names lack the "chr" prefix of
#              its seq files and are not in R's sorted order, holding the tracks and interval
#              sets R writes in it (per-chromosome and indexed);
#   tgt/       a target database for liftover;
#   expected/  R's outputs on them, one .rds per output.
# R sorts the chromosomes of such a database with order(), which depends on the collation:
# run it in the lab's locale (en_US.UTF-8, R built with ICU).
#
#   Rscript make_chromdb.R <out_dir> [misha_lib]
args <- commandArgs(TRUE)
out <- args[1]
if (length(args) > 1) library(misha, lib.loc = args[2]) else library(misha)
stopifnot(capabilities("ICU"), Sys.getlocale("LC_COLLATE") != "C")
options(gmax.processes = 1)

set.seed(17)
dir.create(out, showWarnings = FALSE)
expected <- file.path(out, "expected")
dir.create(expected, showWarnings = FALSE)
save_expected <- function(obj, id) saveRDS(obj, file.path(expected, paste0(id, ".rds")))

fasta_dir <- tempfile()
dir.create(fasta_dir)
make_db <- function(groot, chroms, prefix) {
    fastas <- file.path(fasta_dir, paste0(prefix, names(chroms), ".fasta"))
    for (i in seq_along(chroms)) {
        seq <- paste(sample(c("A", "C", "G", "T"), chroms[[i]], replace = TRUE), collapse = "")
        writeLines(c(paste0(">", prefix, names(chroms)[i]), seq), fastas[i])
    }
    gdb.create(groot, fastas, format = "per-chromosome")
}

# --- the source database: chrom_sizes.txt unprefixed, out of order ------------------------
# R orders chr1_KI270706v1_random before chr10 (Python's sorted() and a C locale do not).
chroms <- c("2" = 3000, "10" = 2500, "1" = 4000, "X" = 1500, "1_KI270706v1_random" = 1200)
groot <- file.path(out, "db")
make_db(groot, chroms, "chr")
writeLines(paste(names(chroms), chroms, sep = "\t"), file.path(groot, "chrom_sizes.txt"))
gsetroot(groot)

# one value per row, the rows in chrom_sizes.txt order
cs <- paste0("chr", names(chroms))
iv1d <- data.frame(
    chrom = rep(cs, each = 2),
    start = rep(c(100, 500), length(cs)),
    end = rep(c(300, 700), length(cs))
)
vals <- seq_len(nrow(iv1d)) * 1.5
gtrack.create_sparse("sparse", "", iv1d, vals)
gtrack.create_dense("dense", "", iv1d, vals, 100, NaN)
array_src <- file.path(fasta_dir, "array.tsv")
write.table(cbind(iv1d, c1 = vals, c2 = 10 * vals), array_src, sep = "\t", quote = FALSE, row.names = FALSE)
gtrack.array.import("array", "", array_src)

# 2D: pairs across chromosomes, listed out of order
iv2d <- data.frame(
    chrom1 = c("chr10", "chr1", "chr2", "chrX", "chr1_KI270706v1_random", "chr1"),
    start1 = c(100, 200, 300, 400, 500, 600),
    end1 = c(150, 260, 380, 470, 590, 650),
    chrom2 = c("chrX", "chr2", "chr2", "chr1", "chr10", "chr1_KI270706v1_random"),
    start2 = c(50, 60, 70, 80, 90, 100),
    end2 = c(90, 120, 110, 130, 150, 170)
)
vals2d <- c(10, 20, 30, 40, 50, 60)
gtrack.2d.create("rects", "", iv2d, vals2d)
points_src <- file.path(fasta_dir, "points.tsv")
points <- iv2d
points$end1 <- points$start1 + 1
points$end2 <- points$start2 + 1
write.table(cbind(points, value = vals2d), points_src, sep = "\t", quote = FALSE, row.names = FALSE)
gtrack.2d.import("points", "", points_src)

# the same tracks, indexed
for (t in c("sparse", "dense", "array")) {
    gtrack.copy(t, paste0(t, "_idx"))
    gtrack.convert_to_indexed(paste0(t, "_idx"))
}
for (t in c("rects", "points")) {
    gtrack.copy(t, paste0(t, "_idx"))
    gtrack.2d.convert_to_indexed(paste0(t, "_idx"), remove.old = TRUE)
}

# interval sets: small, big (one file per chromosome) and indexed big
gintervals.save("iv1d", iv1d)
gintervals.save("iv2d", iv2d)
old_big <- getOption("gbig.intervals.size")
options(gbig.intervals.size = 2)
gintervals.save("big1d", iv1d)
gintervals.save("big2d", iv2d)
gintervals.save("big1d_idx", iv1d)
gintervals.save("big2d_idx", iv2d)
options(gbig.intervals.size = old_big)
gintervals.convert_to_indexed("big1d_idx", remove.old = TRUE)
# the .meta gintervals.update(NULL) of chr1 leaves in big1d (its index, once converted, keeps chr1)
f <- file(file.path(groot, "tracks", "big1d.interv", ".meta"), "rb")
meta <- unserialize(f)
close(f)
meta_dir <- tempfile()
dir.create(meta_dir)
misha:::.gintervals.big.save_meta(meta_dir, meta$stats[as.character(meta$stats$chrom) != "chr1", ], meta$zeroline)
file.copy(file.path(meta_dir, ".meta"), file.path(out, "big1d_without_chr1.meta"), overwrite = TRUE)
gintervals.2d.convert_to_indexed("big2d_idx", remove.old = TRUE)

# --- outputs ---------------------------------------------------------------------------
save_expected(gintervals.all(), "allgenome")
save_expected(gintervals.2d.all(), "allgenome2d")
for (t in c("dense", "dense_idx")) {
    save_expected(gextract(t, gintervals.all(), iterator = 100), paste0("extract_", t))
}
for (t in c("sparse", "sparse_idx", "array", "array_idx")) {
    save_expected(gextract(t, gintervals.all()), paste0("extract_", t))
}
for (t in c("rects", "points", "rects_idx", "points_idx")) {
    save_expected(gextract(t, gintervals.2d.all()), paste0("extract_", t))
}
save_expected(gextract("sparse", gintervals.all(), iterator = "iv1d"), "extract_sparse_iter_set")
save_expected(gscreen("dense > 5", gintervals.all()), "screen_dense")
save_expected(giterator.intervals(intervals = gintervals.all(), iterator = 1000), "iterator_1000")
for (s in c("iv1d", "iv2d", "big1d", "big2d", "big1d_idx", "big2d_idx")) {
    save_expected(gintervals.load(s), paste0("load_", s))
}
save_expected(gintervals.load("big1d", chrom = "chr10"), "load_big1d_chr10")
save_expected(gintervals.load("big2d", chrom1 = "chr1", chrom2 = "chr2"), "load_big2d_chr1_chr2")

mixed <- gintervals(
    c("chrX", "chr10", "chr1_KI270706v1_random", "chr1", "chr2", "chr1"),
    c(0, 100, 50, 900, 10, 950),
    c(200, 300, 400, 1000, 20, 1100)
)
other <- gintervals(c("chr1", "chr10", "chr2", "chr1_KI270706v1_random"), c(950, 0, 0, 0), c(2000, 150, 15, 60))
save_expected(gintervals.canonic(mixed), "canonic")
save_expected(gintervals.union(mixed, other), "union")
save_expected(gintervals.intersect(gintervals.canonic(mixed), other), "intersect")
save_expected(gintervals.diff(mixed, other), "diff")

f <- tempfile()
gextract("sparse", gintervals.all(), file = f)
save_expected(readLines(f), "extract_file_lines")
gtrack.export_bedgraph("sparse", f)
save_expected(readLines(f), "bedgraph_lines")
gdb.export_fasta(f, overwrite = TRUE)
save_expected(grep("^>", readLines(f), value = TRUE), "fasta_headers")

# --- liftover from the indexed and per-chromosome tracks into tgt -------------------------
tgt <- file.path(out, "tgt")
make_db(tgt, c("T1" = 8000, "T2" = 8000), "chr")
chain_file <- file.path(out, "chain.txt")
chain_rows <- list(
    c("chr1", 4000, "chrT1", 0), c("chr10", 2500, "chrT1", 4000), c("chr2", 3000, "chrT2", 0),
    c("chrX", 1500, "chrT2", 3000), c("chr1_KI270706v1_random", 1200, "chrT2", 5000)
)
lines <- character(0)
for (i in seq_along(chain_rows)) {
    r <- chain_rows[[i]]
    size <- as.numeric(r[2])
    tstart <- as.numeric(r[4])
    lines <- c(
        lines,
        sprintf("chain 1000 %s %d + 0 %d %s 8000 + %d %d %d", r[1], size, size, r[3], tstart, tstart + size, i),
        sprintf("%d", size), ""
    )
}
writeLines(lines, chain_file)
# the same chain, its source chromosomes named as in chrom_sizes.txt
writeLines(sub("^chain 1000 chr", "chain 1000 ", lines), file.path(out, "chain_unprefixed.txt"))
src_tracks <- file.path(groot, "tracks")
gsetroot(tgt)
chain <- gintervals.load_chain(chain_file)
for (t in c("sparse", "sparse_idx", "dense", "dense_idx", "rects", "rects_idx", "points", "points_idx")) {
    lifted <- paste0("lifted_", t)
    gtrack.liftover(lifted, "", file.path(src_tracks, paste0(t, ".track")), chain)
    all <- if (t %in% c("rects", "rects_idx", "points", "points_idx")) gintervals.2d.all() else gintervals.all()
    it <- if (startsWith(t, "dense")) 100 else NULL
    save_expected(gextract(lifted, all, iterator = it), paste0("liftover_", t))
    gtrack.rm(lifted, force = TRUE)
}
chain <- gintervals.load_chain(file.path(out, "chain_unprefixed.txt"))
for (t in c("sparse", "sparse_idx")) {
    lifted <- paste0("lifted_", t)
    gtrack.liftover(lifted, "", file.path(src_tracks, paste0(t, ".track")), chain)
    save_expected(gextract(lifted, gintervals.all()), paste0("liftover_unprefixed_chain_", t))
    gtrack.rm(lifted, force = TRUE)
}

# R's session caches are not part of the fixture (git keeps no empty directories either:
# the test recreates tgt/tracks)
for (d in c(groot, tgt)) {
    unlink(file.path(d, c(".db.cache", ".db.cache.dirty")))
}
