# Regenerate tests/test_data/liftover_2d from R misha (5.12.1 or later).
#
# tests/test_liftover_2d_r_bytes.py checks that gtrack_liftover of a 2D track
# writes the very bytes R's gtrack.liftover writes. It compares against FROZEN
# R output, so it runs in ordinary CI with no R installed: this script builds
# the source tracks with R (gzipped into the fixture, since the read order of a
# source follows its quad tree, which R and pymisha lay out differently), writes
# the chains, lifts every case with R and records the size and SHA-256 of each
# file of each lifted track in baseline.json. Re-run only when the cases change:
#
#   Rscript tools/generate_liftover_2d_baseline.R tests/test_data/liftover_2d

suppressMessages(library(misha))
suppressMessages(library(jsonlite))
suppressMessages(library(digest))

out <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(out))
unlink(out, recursive = TRUE)
dir.create(out, recursive = TRUE)
out <- normalizePath(out)
work <- tempfile("liftover_2d_")
dir.create(work)

# A database of the given chromosomes. gdb.create builds an indexed database only
# from a single multi-FASTA file.
mkdb <- function(dir, chroms, sizes, format) {
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

set.seed(19)
src_chroms <- c("chrS1", "chrS2")
src_sizes <- c(2000, 1500)

# Points: contacts packed into 40bp windows that the "keep" chain lands on one target
# window, so that target points collect several contributors, plus contacts spread
# over chrS1 [0, 500), which that chain maps to two target blocks.
win <- data.frame(chrom = c("chrS1", "chrS1", "chrS2", "chrS2"), start = c(0, 1000, 0, 960))
ends <- function(k) {
    w <- win[sample(nrow(win), k, replace = TRUE), ]
    data.frame(chrom = w$chrom, start = w$start + sample(0:39, k, replace = TRUE))
}
e1 <- ends(700)
e2 <- ends(700)
contacts <- data.frame(chrom1 = e1$chrom, start1 = e1$start, chrom2 = e2$chrom, start2 = e2$start)
contacts <- rbind(contacts, data.frame(chrom1 = "chrS1", start1 = sample(0:499, 700, replace = TRUE),
                                       chrom2 = "chrS1", start2 = sample(0:499, 700, replace = TRUE)))
contacts$v <- sample(1:40, nrow(contacts), replace = TRUE) / 4
contacts$v[sample(nrow(contacts), 6)] <- Inf
contacts$v[sample(nrow(contacts), 6)] <- -Inf
# one contact per point and orientation (a cis contact is imported with its mirror)
lo <- contacts$chrom1 == contacts$chrom2 & contacts$start1 > contacts$start2
key <- ifelse(lo, paste(contacts$chrom1, contacts$start2, contacts$start1), paste(contacts$chrom1, contacts$start1, contacts$chrom2, contacts$start2))
contacts <- contacts[!duplicated(key), ]
contacts_file <- file.path(work, "contacts.tsv")
write.table(data.frame(chrom1 = contacts$chrom1, start1 = contacts$start1, end1 = contacts$start1 + 1,
                       chrom2 = contacts$chrom2, start2 = contacts$start2, end2 = contacts$start2 + 1, value = contacts$v),
            contacts_file, sep = "\t", quote = FALSE, row.names = FALSE)

# Rects: disjoint, one per cell of a jittered grid, in every chromosome pair
rects <- NULL
for (c1 in seq_along(src_chroms)) {
    for (c2 in seq_along(src_chroms)) {
        step <- 100
        g <- expand.grid(x = seq(0, src_sizes[c1] - step, by = step), y = seq(0, src_sizes[c2] - step, by = step))
        g <- g[sample(nrow(g), 30), ]
        w <- sample(10:90, nrow(g), replace = TRUE)
        h <- sample(10:90, nrow(g), replace = TRUE)
        x1 <- g$x + vapply(step - w, function(m) sample(0:m, 1), 0)
        y1 <- g$y + vapply(step - h, function(m) sample(0:m, 1), 0)
        rects <- rbind(rects, data.frame(chrom1 = src_chroms[c1], start1 = x1, end1 = x1 + w,
                                         chrom2 = src_chroms[c2], start2 = y1, end2 = y1 + h))
    }
}
rect_vals <- sample(1:40, nrow(rects), replace = TRUE) / 4
rect_vals[sample(nrow(rects), 4)] <- NaN
rect_vals[sample(nrow(rects), 4)] <- Inf

# the source tracks, per chromosome pair and indexed; the fixture holds chrom_sizes.txt
# (which is what locates the database of an indexed track) and the track files, gzipped
for (fmt in c("per-chromosome", "indexed")) {
    db <- mkdb(file.path(work, paste0("src_", fmt)), src_chroms, src_sizes, fmt)
    gdb.init(db)
    suppressMessages(gtrack.2d.import_contacts("pts", "x", contacts_file, fends = NULL))
    gtrack.2d.create("rct", "x", gintervals.2d(rects$chrom1, rects$start1, rects$end1, rects$chrom2, rects$start2, rects$end2), rect_vals)
    fix <- file.path(out, paste0("src_", fmt))
    dir.create(file.path(fix, "tracks"), recursive = TRUE)
    file.copy(file.path(db, "chrom_sizes.txt"), fix)
    for (tr in c("pts", "rct")) {
        tdir <- file.path(fix, "tracks", paste0(tr, ".track"))
        dir.create(tdir)
        for (f in list.files(file.path(db, "tracks", paste0(tr, ".track")))) {
            con <- gzfile(file.path(tdir, paste0(f, ".gz")), "wb")
            writeBin(readBin(file.path(db, "tracks", paste0(tr, ".track"), f), "raw", 1e8), con)
            close(con)
        }
    }
}

write_chain <- function(name, entries) {
    path <- file.path(out, name)
    for (e in entries) {
        cat(sprintf("chain 1000 %s %d %s %d %d %s %d %s %d %d %d\n%d\n\n", e[[1]], e[[2]], e[[3]], e[[4]], e[[5]],
                    e[[6]], e[[7]], e[[8]], e[[9]], e[[10]], e[[11]], e[[5]] - e[[4]]), file = path, append = TRUE)
    }
    path
}
# 1:1 blocks: chrS1 [600, 800) is unmapped, [800, 1400) lands inverted (given on the
# target's reverse strand), [1400, 2000) turns chrS1 cis contacts into chrT1-chrT2 ones
main_chain <- write_chain("main.chain", list(
    list("chrS1", 2000, "+", 0, 600, "chrT1", 8000, "+", 100, 700, 1),
    list("chrS1", 2000, "+", 800, 1400, "chrT1", 8000, "-", 1000, 1600, 2),
    list("chrS1", 2000, "+", 1400, 2000, "chrT2", 6000, "+", 0, 600, 3),
    list("chrS2", 1500, "+", 0, 1500, "chrT2", 6000, "+", 3000, 4500, 4)
))
# blocks of both source chromosomes onto chrT1 [0, 500), one inverted, and chrS1 [0, 500)
# onto two target blocks
keep_chain <- write_chain("keep.chain", list(
    list("chrS1", 2000, "+", 0, 500, "chrT1", 8000, "+", 0, 500, 1),
    list("chrS1", 2000, "+", 1000, 1500, "chrT1", 8000, "+", 0, 500, 2),
    list("chrS2", 1500, "+", 0, 500, "chrT1", 8000, "+", 0, 500, 3),
    list("chrS2", 1500, "+", 500, 1000, "chrT1", 8000, "-", 7500, 8000, 4),
    list("chrS1", 2000, "+", 0, 500, "chrT1", 8000, "+", 2000, 2500, 5),
    list("chrS2", 1500, "+", 1000, 1500, "chrT2", 6000, "+", 0, 500, 6)
))

aggs <- c("mean", "median", "sum", "min", "max", "count", "first", "last", "nth",
          "max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac")
cases <- list()
add <- function(name, track, src, tgt, chain, policy = "auto", agg = "mean", nth = NA, min_n = NA, na_rm = TRUE, max_data_size = 1e7) {
    cases[[length(cases) + 1]] <<- list(name = name, track = track, src = src, tgt = tgt, chain = chain, policy = policy,
                                         agg = agg, nth = nth, min_n = min_n, na_rm = na_rm, max_data_size = max_data_size)
}
for (tr in c("pts", "rct")) {
    add(paste0(tr, "_main"), tr, "per-chromosome", "per-chromosome", "main.chain")
    add(paste0(tr, "_main_indexed"), tr, "indexed", "indexed", "main.chain")
    for (agg in aggs) {
        add(paste0(tr, "_keep_", gsub(".", "_", agg, fixed = TRUE)), tr, "per-chromosome", "per-chromosome", "keep.chain",
            policy = "keep", agg = agg, nth = if (agg == "nth") 2 else NA)
    }
    add(paste0(tr, "_keep_nth3"), tr, "per-chromosome", "per-chromosome", "keep.chain", policy = "keep", agg = "nth", nth = 3)
    add(paste0(tr, "_keep_min_n"), tr, "per-chromosome", "per-chromosome", "keep.chain", policy = "keep", min_n = 2)
    add(paste0(tr, "_keep_na_rm_false"), tr, "per-chromosome", "per-chromosome", "keep.chain", policy = "keep", agg = "sum", na_rm = FALSE)
    add(paste0(tr, "_keep_indexed"), tr, "indexed", "indexed", "keep.chain", policy = "keep", agg = "last")
}
add("pts_keep_agg_policy", "pts", "per-chromosome", "per-chromosome", "keep.chain", policy = "agg", agg = "first")
# pairs split into subtrees
add("pts_main_split", "pts", "per-chromosome", "per-chromosome", "main.chain", max_data_size = 100)
add("pts_keep_split", "pts", "per-chromosome", "per-chromosome", "keep.chain", policy = "keep", agg = "first", max_data_size = 300)
add("pts_keep_split_indexed", "pts", "indexed", "indexed", "keep.chain", policy = "keep", agg = "nth", nth = 2, max_data_size = 300)

tgt_dbs <- list()
for (fmt in c("per-chromosome", "indexed")) tgt_dbs[[fmt]] <- mkdb(file.path(work, paste0("tgt_", fmt)), c("chrT1", "chrT2"), c(8000, 6000), fmt)

for (i in seq_along(cases)) {
    cs <- cases[[i]]
    gdb.init(tgt_dbs[[cs$tgt]])
    src_dir <- file.path(work, paste0("src_", cs$src), "db", "tracks", paste0(cs$track, ".track"))
    args <- list(cs$name, "x", src_dir, file.path(out, cs$chain), src_overlap_policy = "keep",
                 tgt_overlap_policy = cs$policy, multi_target_agg = cs$agg, na.rm = cs$na_rm)
    if (!is.na(cs$nth)) args$params <- cs$nth
    if (!is.na(cs$min_n)) args$min_n <- cs$min_n
    withr::with_options(list(gmax.data.size = cs$max_data_size), suppressMessages(do.call(gtrack.liftover, args)))
    tdir <- file.path(tgt_dbs[[cs$tgt]], "tracks", paste0(cs$name, ".track"))
    files <- sort(list.files(tdir))
    cases[[i]]$files <- lapply(setNames(files, files), function(f) {
        list(size = file.size(file.path(tdir, f)), sha256 = digest(file = file.path(tdir, f), algo = "sha256"))
    })
}

writeLines(toJSON(list(misha = as.character(packageVersion("misha")), cases = cases), auto_unbox = TRUE, pretty = TRUE,
                  digits = NA, na = "null"), file.path(out, "baseline.json"))
unlink(work, recursive = TRUE)
