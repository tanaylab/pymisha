# Regenerate tests/test_data/stats_2d_nan from R misha (5.12.1 or later).
#
# tests/test_2d_nan_stats_r_parity.py checks that pymisha reads 2D tracks with
# NaN cells as R does: R adds a NaN cell to the statistics of any query that
# covers it (area counted, weighted sum and avg NaN), and min / max follow the
# order of R's tree walk. The tracks are R's own files (gzipped into the
# fixture), since that order follows the quad tree, and the values R reads from
# them are frozen in baseline.json, so the test needs no R. It also checks that
# gtrack_2d_create writes R's bytes for rects with NaN values (rct_input.tsv,
# whose SHA-256 per file R wrote is in baseline.json). Re-run only when the
# cases change:
#
#   Rscript tools/generate_2d_nan_stats_baseline.R tests/test_data/stats_2d_nan

suppressMessages(library(misha))
suppressMessages(library(jsonlite))
suppressMessages(library(digest))

out <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(out))
unlink(out, recursive = TRUE)
dir.create(out, recursive = TRUE)
out <- normalizePath(out)
work <- tempfile("stats_2d_nan_")
dir.create(work)

chroms <- c("chrA", "chrB")
sizes <- c(4000, 3000)
fa <- file.path(work, paste0(chroms, ".fasta"))
for (i in seq_along(chroms)) cat(sprintf(">%s\n%s\n", chroms[i], strrep("A", sizes[i])), file = fa[i])
db <- file.path(work, "db")
suppressMessages(gdb.create(groot = db, fasta = fa, format = "per-chromosome"))
gdb.init(db)

set.seed(23)
vals <- function(n) {
    v <- round(runif(n, -5, 5), 3)
    v[sample(n, round(n / 15))] <- NaN
    v
}

# rectangles: disjoint, one per cell of a jittered grid
grid_rects <- function(c1, c2, s1, s2, step) {
    g <- expand.grid(x = seq(0, s1 - step, by = step), y = seq(0, s2 - step, by = step))
    w <- sample(5:(step - 1), nrow(g), replace = TRUE)
    h <- sample(5:(step - 1), nrow(g), replace = TRUE)
    x1 <- g$x + vapply(step - w, function(m) sample(0:m, 1), 0)
    y1 <- g$y + vapply(step - h, function(m) sample(0:m, 1), 0)
    data.frame(chrom1 = c1, start1 = x1, end1 = x1 + w, chrom2 = c2, start2 = y1, end2 = y1 + h)
}
rects <- rbind(grid_rects("chrA", "chrA", 4000, 4000, 160), grid_rects("chrA", "chrB", 4000, 3000, 200))
# a small chunk size puts the tree in many chunks
withr::with_options(list(gtrack.chunk.size = 3000), gtrack.2d.create(
    "rct", "x", gintervals.2d(rects$chrom1, rects$start1, rects$end1, rects$chrom2, rects$start2, rects$end2),
    vals(nrow(rects))
))

# the same rects and values in one chunk, sorted as gtrack_2d_create inserts them
rct_input <- cbind(rects, value = vals(nrow(rects)))
rct_input <- rct_input[order(rct_input$chrom1, rct_input$chrom2, rct_input$start1, rct_input$start2), ]
write.table(transform(rct_input, value = ifelse(is.nan(value), "NaN", sprintf("%.9g", value))),
            file.path(out, "rct_input.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
gtrack.2d.create("rct1", "x", gintervals.2d(rct_input$chrom1, rct_input$start1, rct_input$end1,
                                            rct_input$chrom2, rct_input$start2, rct_input$end2), rct_input$value)
rct1_dir <- file.path(db, "tracks", "rct1.track")
rct1_files <- lapply(setNames(nm = sort(list.files(rct1_dir))), function(f) {
    list(size = file.size(file.path(rct1_dir, f)), sha256 = digest(file = file.path(rct1_dir, f), algo = "sha256"))
})

# points: contacts, a cis contact imported with its mirror
n <- c(900, 500, 300)
pairs <- data.frame(c1 = c("chrA", "chrA", "chrB"), c2 = c("chrA", "chrB", "chrB"), s1 = c(4000, 4000, 3000), s2 = c(4000, 3000, 3000))
contacts <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i) {
    p <- pairs[i, ]
    data.frame(chrom1 = p$c1, start1 = sample(0:(p$s1 - 1), n[i], replace = TRUE),
               chrom2 = p$c2, start2 = sample(0:(p$s2 - 1), n[i], replace = TRUE))
}))
cis <- contacts$chrom1 == contacts$chrom2
key <- paste(contacts$chrom1, ifelse(cis, pmin(contacts$start1, contacts$start2), contacts$start1),
             contacts$chrom2, ifelse(cis, pmax(contacts$start1, contacts$start2), contacts$start2))
contacts <- contacts[!duplicated(key), ]
contacts$value <- vals(nrow(contacts))
contacts_file <- file.path(work, "contacts.tsv")
write.table(data.frame(chrom1 = contacts$chrom1, start1 = contacts$start1, end1 = contacts$start1 + 1,
                       chrom2 = contacts$chrom2, start2 = contacts$start2, end2 = contacts$start2 + 1,
                       value = ifelse(is.nan(contacts$value), "NaN", contacts$value)),
            contacts_file, sep = "\t", quote = FALSE, row.names = FALSE)
withr::with_options(list(gtrack.chunk.size = 3000),
                    suppressMessages(gtrack.2d.import_contacts("pts", "x", contacts_file, fends = NULL)))
# a pair of at least 4 x gmax.data.size points is written in parts (subtrees)
withr::with_options(list(gmax.data.size = 100),
                    suppressMessages(gtrack.2d.import_contacts("pts_split", "x", contacts_file, fends = NULL)))

# the database (chrom_sizes.txt) and the track files, gzipped
file.copy(file.path(db, "chrom_sizes.txt"), out)
for (tr in c("rct", "pts", "pts_split")) {
    tdir <- file.path(out, "tracks", paste0(tr, ".track"))
    dir.create(tdir, recursive = TRUE)
    for (f in list.files(file.path(db, "tracks", paste0(tr, ".track")))) {
        con <- gzfile(file.path(tdir, paste0(f, ".gz")), "wb")
        writeBin(readBin(file.path(db, "tracks", paste0(tr, ".track"), f), "raw", 1e8), con)
        close(con)
    }
}

# queries: whole pairs, quadrants (a node of the tree taken whole), and random
# rectangles from a cell to most of a pair
q <- data.frame(chrom1 = c("chrA", "chrA", "chrB", "chrA", "chrA", "chrA", "chrA"),
                start1 = c(0, 0, 0, 0, 2000, 1000, 0), end1 = c(4000, 4000, 3000, 2000, 4000, 2000, 1000),
                chrom2 = c("chrA", "chrB", "chrB", "chrA", "chrA", "chrA", "chrA"),
                start2 = c(0, 0, 0, 0, 2000, 1000, 0), end2 = c(4000, 3000, 3000, 2000, 4000, 2000, 1000))
for (i in 1:45) {
    p <- pairs[sample(3, 1), ]
    w <- sample(c(1, 7, 40, 150, 600, 1500, 2900), 1)
    h <- sample(c(1, 7, 40, 150, 600, 1500, 2900), 1)
    x1 <- sample(0:(p$s1 - w), 1)
    y1 <- sample(0:(p$s2 - h), 1)
    q <- rbind(q, data.frame(chrom1 = p$c1, start1 = x1, end1 = x1 + w, chrom2 = p$c2, start2 = y1, end2 = y1 + h))
}
queries <- gintervals.2d(q$chrom1, q$start1, q$end1, q$chrom2, q$start2, q$end2)
# a grid of disjoint cells over the pairs, for the functions that summarize an iterator
grid <- giterator.intervals("rct", gintervals.2d.all(), iterator = c(500, 500))[, 1:6]

bands <- list(none = NULL, wide = c(-1500, 1500), lower = c(-400, 0), upper = c(100, 2600), far = c(-3900, -50))
funcs <- c(avg = "avg", min = "min", max = "max", weighted_sum = "weighted.sum", area = "area")
num <- function(x) ifelse(is.nan(x), "NaN", ifelse(is.infinite(x), ifelse(x > 0, "Inf", "-Inf"), sprintf("%.17g", x)))

cases <- list()
for (tr in c("rct", "pts", "pts_split")) {
    for (f in names(funcs)) gvtrack.create(paste0("v_", f), tr, funcs[[f]])
    for (b in names(bands)) {
        band <- bands[[b]]
        # the queries overlap, and an iterator must not: one query at a time
        r <- do.call(rbind, lapply(seq_len(nrow(queries)), function(i) {
            r <- gextract(c(tr, paste0("v_", names(funcs))), queries[i, ], iterator = queries[i, ], band = band,
                          colnames = c("track", names(funcs)))
            if (!is.null(r)) r$intervalID <- i
            r
        }))
        res <- c(lapply(r[c("start1", "end1", "start2", "end2", "intervalID")], as.numeric),
                 list(chrom1 = as.character(r$chrom1), chrom2 = as.character(r$chrom2)),
                 lapply(r[c("track", names(funcs))], num))
        s <- gsummary(tr, gintervals.2d.all(), iterator = grid, band = band)
        qs <- gquantiles(tr, c(0.1, 0.5, 0.9), gintervals.2d.all(), iterator = grid, band = band)
        d <- gdist(tr, c(-5, -2, 0, 2, 5), gintervals.2d.all(), iterator = grid, band = band)
        sc <- gscreen(paste(tr, "> 0"), gintervals.2d.all(), iterator = grid, band = band)
        cases[[length(cases) + 1]] <- list(
            track = tr, band = band, band_name = b, values = res,
            summary = setNames(as.list(num(s)), names(s)),
            quantiles = num(unname(qs)),
            dist = as.integer(d),
            screen = if (is.null(sc)) list() else lapply(as.list(sc[, 1:6]), function(x) if (is.factor(x)) as.character(x) else x)
        )
    }
    for (f in names(funcs)) gvtrack.rm(paste0("v_", f))
}

writeLines(toJSON(list(misha = as.character(packageVersion("misha")), queries = queries, grid = grid, cases = cases,
                       rct1_files = rct1_files),
                  auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), file.path(out, "baseline.json"))
unlink(work, recursive = TRUE)
