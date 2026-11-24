# Regenerate tests/test_data/windows from R misha (the version with gextract.windows).
#
# tests/test_gextract_windows_r_parity.py checks that gseq_onehot and gextract_windows
# return the arrays R's gseq.onehot and gextract.windows return on the example database
# (which pymisha ships too): the inputs of each case, the kept and dropped rows, the
# SHA-256 of the one-hot array in (N, 4, L) row-major order and the signal values are
# frozen in baseline.json, so the test needs no R. Re-run only when the cases change:
#
#   Rscript tools/generate_windows_baseline.R tests/test_data/windows

suppressMessages(library(misha))
suppressMessages(library(jsonlite))
suppressMessages(library(digest))

out <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(!is.na(out))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(gmax.processes = 2)
gdb.init_examples(tempfile("windows_baseline_"))

# the bytes of x in numpy's (N, 4, L) C order
x_sha <- function(x) {
    digest(as.raw(as.integer(aperm(x, c(3, 2, 1)))), algo = "sha256", serialize = FALSE)
}
# y in numpy's (N, T, L') C order, NaN as null
y_values <- function(y) {
    v <- as.numeric(aperm(y, c(3, 2, 1)))
    lapply(v, function(e) if (is.nan(e)) NULL else e)
}

set.seed(2026)
sizes <- gintervals.all()
n <- 200
chroms <- sample(as.character(sizes$chrom), n, replace = TRUE)
lens <- sizes$end[match(chroms, sizes$chrom)]
starts <- floor(runif(n) * lens)
loci <- data.frame(
    chrom = chroms, start = starts, end = pmin(starts + sample(1:600, n, replace = TRUE), lens),
    strand = sample(c(-1, 1), n, replace = TRUE)
)
loci <- loci[loci$end > loci$start, ]
# and some whose windows run off a chromosome end
edge <- c(20, 150, 400)
loci <- rbind(loci, data.frame(
    chrom = rep(as.character(sizes$chrom), each = 2 * length(edge)),
    start = c(sapply(sizes$end, function(l) c(edge, l - edge - 1))),
    end = c(sapply(sizes$end, function(l) c(edge + 1, l - edge))),
    strand = 1
))
exclude <- data.frame(chrom = c("chr1", "chr2", "chrX"), start = c(100000, 50000, 20000), end = c(140000, 60000, 21000))

cases <- list(
    list(
        name = "signals_jitter_exclude_counts",
        args = list(tracks = c("dense_track", "sparse_track"), width = 501, signal_width = 300, jitter = 10,
                    binsize = 20, min_counts = 0.5, count_track = 2, exclude = TRUE)
    ),
    list(name = "per_base_unstranded", args = list(tracks = "dense_track", width = 100, binsize = 1, use_strand = FALSE)),
    list(name = "sequence_only", args = list(width = 1000))
)

res <- list(loci = loci, exclude = exclude, cases = list())
for (case in cases) {
    a <- case$args
    w <- gextract.windows(loci,
        tracks = a$tracks, width = a$width,
        signal_width = if (is.null(a$signal_width)) a$width else a$signal_width,
        jitter = if (is.null(a$jitter)) 0 else a$jitter,
        binsize = if (is.null(a$binsize)) 1 else a$binsize,
        use_strand = if (is.null(a$use_strand)) TRUE else a$use_strand,
        exclude = if (isTRUE(a$exclude)) exclude else NULL,
        min_counts = a$min_counts,
        count_track = if (is.null(a$count_track)) 1 else a$count_track
    )
    res$cases[[length(res$cases) + 1]] <- list(
        name = case$name, args = a,
        x_dim = dim(w$x), x_sha256 = x_sha(w$x),
        y_dim = if (is.null(w$y)) NULL else dim(w$y), y = if (is.null(w$y)) NULL else y_values(w$y),
        index = w$intervals$index - 1, start = w$intervals$start,
        dropped_index = w$dropped$index - 1, dropped_reason = w$dropped$reason
    )
}

# gseq.onehot with pad = TRUE: windows running past both ends of chr1
len1 <- sizes$end[1]
pad_iv <- data.frame(chrom = "chr1", start = c(-300, len1 - 200, 1000), end = c(200, len1 + 300, 1500), strand = c(-1, 1, -1))
xp <- gseq.onehot(pad_iv, pad = TRUE)
res$pad <- list(intervals = pad_iv, x_dim = dim(xp), x_sha256 = x_sha(xp))

writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null", pretty = FALSE), file.path(out, "baseline.json"))
