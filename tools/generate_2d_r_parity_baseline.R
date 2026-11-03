# Regenerate tests/test_data/2d_r_parity from R misha (5.12.1 or later).
#
# tests/test_2d_r_parity.py checks pymisha's 2D functions against what R misha
# returns on pymisha's own test database (tests/testdb/trackdb/test, whose
# chromosomes 1, 2 and X R calls chr1, chr2 and chrX, in the same order). The
# values R returns are frozen in baseline.json, so the test needs no R. Re-run
# only when the cases change:
#
#   Rscript tools/generate_2d_r_parity_baseline.R tests/testdb/trackdb/test tests/test_data/2d_r_parity

suppressMessages(library(misha))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
src_db <- args[1]
out <- args[2]
stopifnot(!is.na(src_db), !is.na(out))
unlink(out, recursive = TRUE)
dir.create(out, recursive = TRUE)
out <- normalizePath(out)

# a scratch copy of the database, so that nothing is written into the source
work <- tempfile("2d_r_parity_")
dir.create(work)
file.copy(normalizePath(src_db), work, recursive = TRUE)
db <- file.path(work, basename(normalizePath(src_db)))
gdb.init(db)
options(gmultitasking = FALSE)

num <- function(x) ifelse(is.nan(x) | is.na(x), "NaN", ifelse(is.infinite(x), ifelse(x > 0, "Inf", "-Inf"), sprintf("%.17g", x)))
frame <- function(r, value_cols = character()) {
    if (is.null(r)) {
        return(list())
    }
    res <- list(chrom1 = as.character(r$chrom1), start1 = r$start1, end1 = r$end1,
                chrom2 = as.character(r$chrom2), start2 = r$start2, end2 = r$end2)
    for (col in value_cols) res[[col]] <- num(r[[col]])
    if ("intervalID" %in% colnames(r)) res$intervalID <- r$intervalID
    res
}

baseline <- list(misha = as.character(packageVersion("misha")))

# --- 2D statistics of a query that meets no cell --------------------------------------------
# A query in a pair with a file but no cell under it, one in a pair with no file (chr2-chr2),
# one covering cells, and vtracks shifted so that the query stays in range (area 0 or more) or
# leaves the chromosome (out of range: every function NaN, area included).
empty_q <- data.frame(
    chrom1 = c("chr1", "chr2", "chr1", "chr1", "chrX"),
    start1 = c(0, 0, 0, 1000, 0), end1 = c(10, 10, 500000, 30000, 200000),
    chrom2 = c("chr1", "chr2", "chr2", "chr1", "chrX"),
    start2 = c(0, 0, 0, 1000, 0), end2 = c(10, 10, 300000, 400000, 200000)
)
funcs <- c(avg = "avg", min = "min", max = "max", weighted_sum = "weighted.sum", area = "area")
for (f in names(funcs)) gvtrack.create(paste0("v_", f), "rects_track", funcs[[f]])
gvtrack.create("v_area_out", "rects_track", "area")
gvtrack.iterator.2d("v_area_out", sshift1 = 600000, eshift1 = 600000)
gvtrack.create("v_area_in", "rects_track", "area")
gvtrack.iterator.2d("v_area_in", sshift1 = -5, eshift1 = 5)
empty_cols <- c("rects_track", paste0("v_", names(funcs)), "v_area_out", "v_area_in")
baseline$empty_query <- list(
    queries = frame(empty_q),
    cols = empty_cols,
    none = frame(gextract(empty_cols, empty_q, iterator = empty_q), empty_cols),
    band = frame(gextract(empty_cols, empty_q, iterator = empty_q, band = c(-5, 5)), empty_cols)
)
gvtrack.rm("v_area_out")
gvtrack.rm("v_area_in")

# --- 2D iterators: the cells R walks, in its order, and their intervalID ---------------------
# A scope of many disjoint rects per pair, given with the pairs interleaved (R sorts it by
# pair with std::sort, which reorders the rects of a pair, and holds a pair's rects in a quad
# tree, which returns them in its own order once a pair has more than 20), and an iterator of
# disjoint rects given the same way.
set.seed(5)
grid_rects <- function(c1, c2, s1, s2, n1, n2) {
    xs <- seq(0, s1, length.out = n1 + 1)
    ys <- seq(0, s2, length.out = n2 + 1)
    g <- expand.grid(i = 1:n1, j = 1:n2)
    data.frame(chrom1 = c1, start1 = floor(xs[g$i]), end1 = floor(xs[g$i + 1]) - sample(0:3, nrow(g), TRUE),
               chrom2 = c2, start2 = floor(ys[g$j]), end2 = floor(ys[g$j + 1]) - sample(0:3, nrow(g), TRUE))
}
walk_scope <- rbind(grid_rects("chr1", "chr1", 500000, 500000, 6, 5), grid_rects("chr1", "chr2", 500000, 300000, 5, 3),
                    grid_rects("chr1", "chrX", 500000, 200000, 2, 2))
walk_scope <- walk_scope[sample(nrow(walk_scope)), ]
rownames(walk_scope) <- NULL
walk_iter <- rbind(grid_rects("chr1", "chr1", 500000, 500000, 4, 7), grid_rects("chr1", "chr2", 500000, 300000, 3, 4))
walk_iter <- walk_iter[sample(nrow(walk_iter)), ]
rownames(walk_iter) <- NULL
centers <- data.frame(chrom = c("chr1", "chr1", "chr2"), start = c(100000, 300000, 150000), end = c(100001, 300001, 150001))
cgrid <- giterator.cartesian_grid(centers, c(-40000, 0, 40000))
walk_band <- c(-200000, 50000)
gvtrack.create("v_area", "rects_track", "area")
walk_cols <- c("rects_track", "v_area")
walk <- list(scope = frame(walk_scope), iterator = frame(walk_iter), centers = list(chrom = as.character(centers$chrom), start = centers$start, end = centers$end),
             expansion = c(-40000, 0, 40000), band = walk_band)
walk$gextract <- list(
    none = frame(gextract("rects_track", walk_scope), "rects_track"),
    none_band = frame(gextract("rects_track", walk_scope, band = walk_band), "rects_track"),
    track = frame(gextract(walk_cols, walk_scope, iterator = "rects_track"), walk_cols),
    track_band = frame(gextract(walk_cols, walk_scope, iterator = "rects_track", band = walk_band), walk_cols),
    intervals = frame(gextract(walk_cols, walk_scope, iterator = walk_iter), walk_cols),
    intervals_band = frame(gextract(walk_cols, walk_scope, iterator = walk_iter, band = walk_band), walk_cols),
    rect = frame(gextract(walk_cols, walk_scope, iterator = c(70000, 90000)), walk_cols),
    rect_band = frame(gextract(walk_cols, walk_scope, iterator = c(70000, 90000), band = walk_band), walk_cols),
    cartesian = frame(gextract(walk_cols, walk_scope, iterator = cgrid), walk_cols)
)
walk$gscreen <- list(
    none = frame(gscreen("rects_track > 5000", walk_scope)),
    none_band = frame(gscreen("rects_track > 5000", walk_scope, band = walk_band)),
    intervals = frame(gscreen("rects_track > 5000", walk_scope, iterator = walk_iter)),
    rect = frame(gscreen("v_area > 100000", walk_scope, iterator = c(70000, 90000)))
)
walk$giterator <- list(
    track = frame(giterator.intervals("rects_track", walk_scope)),
    intervals = frame(giterator.intervals(intervals = walk_scope, iterator = walk_iter)),
    rect = frame(giterator.intervals(intervals = walk_scope, iterator = c(70000, 90000)))
)
# a virtual track of a 2D track with no iterator iterates the objects of its source
gvtrack.create("v_avg", "rects_track", "avg")
walk$vtrack <- list(
    none = frame(gextract(c("v_area", "v_avg"), walk_scope), c("v_area", "v_avg")),
    none_band = frame(gextract(c("v_area", "v_avg"), walk_scope, band = walk_band), c("v_area", "v_avg"))
)
gvtrack.rm("v_avg")
gvtrack.rm("v_area")
baseline$walk <- walk

# --- functions whose default scope is the 2D genome for a 2D expression -------------------------
cor_detail <- function(r) as.list(setNames(num(as.numeric(r)), colnames(r)))
cis_src <- data.frame(chrom = "chr1", start = c(0, 100000), end = c(50000, 150000))
cis_dom <- data.frame(chrom = "chr1", start = c(0, 250000), end = c(250000, 500000))
cis_breaks <- c(0, 50000, 100000, 200000, 500000)
cis_scope <- data.frame(chrom1 = "chr1", start1 = c(0, 30000), end1 = c(30000, 500000), chrom2 = "chr1", start2 = 0, end2 = 500000)
cis <- function(...) {
    r <- gcis_decay("rects_track", cis_breaks, cis_src, cis_dom, ...)
    list(intra = as.numeric(r[, 1]), inter = as.numeric(r[, 2]))
}
bins_breaks <- c(0, 10000, 70000)
set.seed(17)
baseline$default_scope <- list(
    gcor = cor_detail(gcor("rects_track", "rects_track * 2", details = TRUE)),
    gcor_spearman = cor_detail(gcor("rects_track", "rects_track * 2", method = "spearman.exact", details = TRUE)),
    gcor_rect_band = cor_detail(gcor("rects_track", "rects_track * 2", iterator = c(100000, 100000), band = c(-200000, 0), details = TRUE)),
    gbins_summary = lapply(as.data.frame(unclass(gbins.summary("rects_track", bins_breaks, expr = "rects_track"))), num),
    gbins_quantiles = num(as.numeric(gbins.quantiles("rects_track", bins_breaks, expr = "rects_track", percentiles = c(0.2, 0.5)))),
    gsample_all = sort(gsample("rects_track", 100000)),
    cis_src = list(chrom = "chr1", start = cis_src$start, end = cis_src$end),
    cis_dom = list(chrom = "chr1", start = cis_dom$start, end = cis_dom$end),
    cis_breaks = cis_breaks, cis_scope = frame(cis_scope),
    gcis_decay = cis(), gcis_decay_band = cis(band = c(-300000, 0)), gcis_decay_scope = cis(intervals = cis_scope),
    gcis_decay_rect = cis(iterator = c(20000, 20000))
)

# --- a computed 2D track, read through R's Computer2D ----------------------------------------
# A test-computer track (some pairs without a file); a query that meets part of an object reads
# the computer's value of that part, one that meets no cell area 0 and NaN. Its files are
# gzipped into computed/.
set.seed(3)
misha:::.gtrack.create_test_computer2d("computed2d", prob.skip.chrom = 0.4, max.rect = 300, max.rect.size = 20000)
comp_dir <- file.path(db, "tracks", "computed2d.track")
dir.create(file.path(out, "computed"))
for (f in setdiff(list.files(comp_dir), list.files(comp_dir, pattern = "^\\."))) {
    con <- gzfile(file.path(out, "computed", paste0(f, ".gz")), "wb")
    writeBin(readBin(file.path(comp_dir, f), "raw", 1e8), con)
    close(con)
}
comp_q <- data.frame(chrom1 = c("chr1", "chr1", "chr2", "chr1", "chrX", "chr2"), start1 = c(0, 100000, 0, 400000, 0, 0),
                     end1 = c(100000, 300000, 300000, 400100, 200000, 10),
                     chrom2 = c("chr1", "chr1", "chr2", "chr2", "chr1", "chrX"), start2 = c(0, 50000, 0, 0, 0, 0),
                     end2 = c(100000, 400000, 300000, 300000, 500000, 10))
for (f in names(funcs)) gvtrack.create(paste0("c_", f), "computed2d", funcs[[f]])
comp_cols <- c("computed2d", paste0("c_", names(funcs)))
comp_band <- c(-50000, 50000)
baseline$computed <- list(
    queries = frame(comp_q), cols = comp_cols, band = comp_band,
    vt = frame(gextract(comp_cols, comp_q, iterator = comp_q), comp_cols),
    vt_band = frame(gextract(comp_cols, comp_q, iterator = comp_q, band = comp_band), comp_cols),
    none = frame(gextract("computed2d", comp_q), "computed2d"),
    none_band = frame(gextract("computed2d", comp_q, band = comp_band), "computed2d"),
    rect = frame(gextract("computed2d", comp_q, iterator = c(50000, 50000)), "computed2d")
)
for (f in names(funcs)) gvtrack.rm(paste0("c_", f))

# --- 2D track writers: the bytes of each file R writes ----------------------------------------
# gtrack.2d.create from intervals given with the pairs interleaved (R sorts them by pair with
# std::sort, which reorders the intervals of a pair) and large enough to fill several chunks,
# and from 1 x 1 intervals (still a RECTS track); gtrack.2d.import of rects and of points from
# two files, a pair written whole and in parts (gmax.data.size); gtrack.2d.import_contacts of
# cis and trans contacts with duplicates, whole and in parts. The inputs are saved next to the
# baseline.
suppressMessages(library(digest))
file_hashes <- function(track) {
    dir <- file.path(db, "tracks", paste0(track, ".track"))
    fs <- sort(setdiff(list.files(dir), list.files(dir, pattern = "^\\.")))
    setNames(lapply(fs, function(f) list(size = file.size(file.path(dir, f)), sha256 = digest(file = file.path(dir, f), algo = "sha256"))), fs)
}
write_input <- function(df, name) {
    df$value <- ifelse(is.nan(df$value), "NaN", sprintf("%.9g", df$value))
    write.table(df, file.path(out, name), sep = "\t", quote = FALSE, row.names = FALSE)
}
set.seed(11)
vals <- function(n) {
    v <- round(runif(n, -100, 100), 4)
    v[sample(n, max(1, n %/% 20))] <- NaN
    v
}
create_in <- rbind(grid_rects("chr1", "chr1", 500000, 500000, 40, 40), grid_rects("chr1", "chr2", 500000, 300000, 30, 20),
                   grid_rects("chrX", "chr1", 200000, 500000, 10, 10))
create_in <- create_in[sample(nrow(create_in)), ]
create_in$value <- vals(nrow(create_in))
write_input(create_in, "create_rects.tsv")
gtrack.2d.create("w_create", "x", create_in[, 1:6], create_in$value)
pts_in <- data.frame(chrom1 = "chr1", start1 = sample(0:499999, 300), chrom2 = c("chr1", "chr2")[sample(2, 300, TRUE)],
                     start2 = sample(0:199999, 300))
pts_in <- pts_in[!duplicated(pts_in[, c("chrom1", "start1", "chrom2", "start2")]), ]
pts_in <- data.frame(chrom1 = pts_in$chrom1, start1 = pts_in$start1, end1 = pts_in$start1 + 1,
                     chrom2 = pts_in$chrom2, start2 = pts_in$start2, end2 = pts_in$start2 + 1, value = vals(nrow(pts_in)))
write_input(pts_in, "create_points.tsv")
gtrack.2d.create("w_create_pts", "x", pts_in[, 1:6], pts_in$value)
rect_in <- rbind(grid_rects("chr1", "chr1", 500000, 500000, 20, 20), grid_rects("chr2", "chrX", 300000, 200000, 8, 6))
rect_in <- rect_in[sample(nrow(rect_in)), ]
rect_in$value <- vals(nrow(rect_in))
write_input(rect_in[1:200, ], "import_rects_a.tsv")
write_input(rect_in[201:nrow(rect_in), ], "import_rects_b.tsv")
gtrack.2d.import("w_import", "x", file.path(out, c("import_rects_a.tsv", "import_rects_b.tsv")))
withr::with_options(list(gmax.data.size = 20), gtrack.2d.import("w_import_split", "x", file.path(out, c("import_rects_a.tsv", "import_rects_b.tsv"))))
write_input(pts_in[1:150, ], "import_points_a.tsv")
write_input(pts_in[151:nrow(pts_in), ], "import_points_b.tsv")
gtrack.2d.import("w_import_pts", "x", file.path(out, c("import_points_a.tsv", "import_points_b.tsv")))
withr::with_options(list(gmax.data.size = 10), gtrack.2d.import("w_import_pts_split", "x", file.path(out, c("import_points_a.tsv", "import_points_b.tsv"))))
n <- 600
contacts <- data.frame(chrom1 = c("chr1", "chr2", "chrX")[sample(3, n, TRUE)], chrom2 = c("chr1", "chr2", "chrX")[sample(3, n, TRUE)])
csize <- c(chr1 = 500000, chr2 = 300000, chrX = 200000)
contacts$start1 <- floor(runif(n) * (csize[contacts$chrom1] - 10))
contacts$start2 <- floor(runif(n) * (csize[contacts$chrom2] - 10))
dup <- sample(n, 40)
contacts[dup[1:20], c("chrom1", "start1", "chrom2", "start2")] <- contacts[dup[21:40], c("chrom2", "start2", "chrom1", "start1")]
contacts <- data.frame(chrom1 = contacts$chrom1, start1 = contacts$start1, end1 = contacts$start1 + sample(1:9, n, TRUE),
                       chrom2 = contacts$chrom2, start2 = contacts$start2, end2 = contacts$start2 + sample(1:9, n, TRUE),
                       value = round(runif(n, 0, 10), 3))
write_input(contacts[1:300, ], "contacts_a.tsv")
write_input(contacts[301:n, ], "contacts_b.tsv")
suppressMessages(gtrack.2d.import_contacts("w_contacts", "x", file.path(out, c("contacts_a.tsv", "contacts_b.tsv")), fends = NULL))
withr::with_options(list(gmax.data.size = 16), suppressMessages(
    gtrack.2d.import_contacts("w_contacts_split", "x", file.path(out, c("contacts_a.tsv", "contacts_b.tsv")), fends = NULL)))
baseline$writers <- list(
    create = file_hashes("w_create"), create_points = file_hashes("w_create_pts"),
    import = file_hashes("w_import"), import_split = file_hashes("w_import_split"),
    import_points = file_hashes("w_import_pts"), import_points_split = file_hashes("w_import_pts_split"),
    contacts = file_hashes("w_contacts"), contacts_split = file_hashes("w_contacts_split"),
    types = list(create = gtrack.info("w_create")$type, create_points = gtrack.info("w_create_pts")$type,
                 import_points = gtrack.info("w_import_pts")$type)
)

writeLines(toJSON(baseline, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), file.path(out, "baseline.json"))
unlink(work, recursive = TRUE)
