#!/usr/bin/env Rscript

# Benchmark suite for development R misha.
# Mirrors case IDs from run_pymisha_bench.py.

parse_args <- function(argv) {
    opts <- list(
        rmisha_src = Sys.getenv("RMISHA_SRC", "~/src/misha"),
        db_root = Sys.getenv("MISHA_BENCH_DB", "~/src/pymisha/tests/testdb/trackdb/test"),
        warmup = 1L,
        reps = 5L,
        name_prefix = "bench",
        output_csv = NULL,
        quiet = FALSE
    )

    i <- 1L
    while (i <= length(argv)) {
        key <- argv[[i]]
        if (key == "--rmisha-src") {
            i <- i + 1L
            opts$rmisha_src <- argv[[i]]
        } else if (key == "--db-root") {
            i <- i + 1L
            opts$db_root <- argv[[i]]
        } else if (key == "--warmup") {
            i <- i + 1L
            opts$warmup <- as.integer(argv[[i]])
        } else if (key == "--reps") {
            i <- i + 1L
            opts$reps <- as.integer(argv[[i]])
        } else if (key == "--name-prefix") {
            i <- i + 1L
            opts$name_prefix <- argv[[i]]
        } else if (key == "--output-csv") {
            i <- i + 1L
            opts$output_csv <- argv[[i]]
        } else if (key == "--quiet") {
            opts$quiet <- TRUE
        } else {
            stop(sprintf("Unknown argument: %s", key))
        }
        i <- i + 1L
    }

    opts
}

count_rows <- function(result) {
    if (is.null(result)) {
        return(0L)
    }
    if (is.data.frame(result)) {
        return(as.integer(nrow(result)))
    }
    if (is.matrix(result)) {
        return(as.integer(nrow(result)))
    }
    if (is.vector(result) || is.list(result)) {
        return(as.integer(length(result)))
    }
    NA_integer_
}

bench_callable <- function(fn, warmup, reps) {
    if (warmup > 0L) {
        for (i in seq_len(warmup)) {
            suppressWarnings(invisible(fn()))
        }
    }

    times <- numeric(reps)
    result_rows <- NA_integer_
    for (i in seq_len(reps)) {
        gc()
        t0 <- proc.time()[["elapsed"]]
        result <- suppressWarnings(fn())
        elapsed <- proc.time()[["elapsed"]] - t0
        times[[i]] <- elapsed
        if (i == 1L) {
            result_rows <- count_rows(result)
        }
    }

    list(
        median_s = as.numeric(median(times)),
        std_s = as.numeric(if (reps > 1L) sd(times) else 0.0),
        min_s = as.numeric(min(times)),
        max_s = as.numeric(max(times)),
        result_rows = result_rows
    )
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
opts$rmisha_src <- normalizePath(path.expand(opts$rmisha_src), mustWork = TRUE)
opts$db_root <- normalizePath(path.expand(opts$db_root), mustWork = TRUE)

suppressPackageStartupMessages(library(devtools))
load_all(opts$rmisha_src, quiet = TRUE, export_all = FALSE)

gdb.init(opts$db_root)
options(gmultitasking = FALSE)

vtrack_specs <- list(
    list(base_name = "vt_sum_dense", func = "sum", src = "dense_track", source_density = "dense", threshold = 0.30),
    list(base_name = "vt_sum_sparse", func = "sum", src = "sparse_track", source_density = "sparse", threshold = 0.40),
    list(base_name = "vt_avg_dense", func = "avg", src = "dense_track", source_density = "dense", threshold = 0.08),
    list(base_name = "vt_avg_sparse", func = "avg", src = "sparse_track", source_density = "sparse", threshold = 0.40),
    list(base_name = "vt_global_percentile_dense", func = "global.percentile", src = "dense_track", source_density = "dense", threshold = 0.50),
    list(base_name = "vt_pwm", func = "pwm", src = NULL, source_density = "sequence", threshold = 2.00)
)

profile_specs <- list(
    list(case_suffix = "single_small_dense_iter", chroms = c("1"), start = 0L, end = 50000L, iterator = 100L, iterator_density = "dense", chrom_mode = "single", size_label = "small"),
    list(case_suffix = "single_full_sparse_iter", chroms = c("1"), start = 0L, end = -1L, iterator = 5000L, iterator_density = "sparse", chrom_mode = "single", size_label = "large"),
    list(case_suffix = "multi_medium_dense_iter", chroms = c("1", "2", "X"), start = 0L, end = 100000L, iterator = 200L, iterator_density = "dense", chrom_mode = "multi", size_label = "medium"),
    list(case_suffix = "multi_full_sparse_iter", chroms = c("1", "2", "X"), start = 0L, end = -1L, iterator = 10000L, iterator_density = "sparse", chrom_mode = "multi", size_label = "large")
)

operations <- c("gextract", "gscreen", "gsummary", "gquantiles")
quantiles <- c(0.25, 0.5, 0.75)

run_tag <- format(Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC")
timestamp_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

pssm <- matrix(c(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0
), ncol = 4, byrow = TRUE)
colnames(pssm) <- c("A", "C", "G", "T")

runtime_names <- list()
creation_errors <- list()
for (spec in vtrack_specs) {
    runtime_name <- paste(opts$name_prefix, spec$base_name, run_tag, sep = "_")
    ok <- TRUE
    err <- ""
    tryCatch({
        if (identical(spec$func, "pwm")) {
            gvtrack.create(
                runtime_name,
                NULL,
                func = "pwm",
                pssm = pssm,
                bidirect = TRUE,
                prior = 0.01,
                extend = TRUE
            )
        } else {
            gvtrack.create(runtime_name, spec$src, func = spec$func)
        }
    }, error = function(e) {
        ok <<- FALSE
        err <<- sprintf("%s: %s", class(e)[1], conditionMessage(e))
    })

    if (ok) {
        runtime_names[[spec$base_name]] <- runtime_name
    } else {
        creation_errors[[spec$base_name]] <- err
    }
}

interval_cache <- list()
for (profile in profile_specs) {
    chrom_arg <- if (length(profile$chroms) == 1L) profile$chroms[[1L]] else profile$chroms
    interval_cache[[profile$case_suffix]] <- gintervals(chrom_arg, profile$start, profile$end)
}

rows <- list()
append_row <- function(...) {
    rows[[length(rows) + 1L]] <<- data.frame(..., stringsAsFactors = FALSE)
}

run_case <- function(operation, vtrack_name, threshold, intervals, iterator) {
    if (identical(operation, "gextract")) {
        return(gextract(vtrack_name, intervals, iterator = iterator))
    }
    if (identical(operation, "gscreen")) {
        expr <- sprintf("%s > %.8f", vtrack_name, threshold)
        return(gscreen(expr, intervals, iterator = iterator))
    }
    if (identical(operation, "gsummary")) {
        return(gsummary(vtrack_name, intervals, iterator = iterator))
    }
    if (identical(operation, "gquantiles")) {
        return(gquantiles(vtrack_name, quantiles, intervals, iterator = iterator))
    }
    stop(sprintf("Unsupported operation: %s", operation))
}

total_cases <- length(operations) * length(vtrack_specs) * length(profile_specs)
case_idx <- 0L

for (operation in operations) {
    for (spec in vtrack_specs) {
        for (profile in profile_specs) {
            case_idx <- case_idx + 1L
            case_id <- paste(operation, spec$base_name, profile$case_suffix, sep = "__")
            runtime_name <- runtime_names[[spec$base_name]]
            status <- "success"
            err <- ""
            median_s <- NA_real_
            std_s <- NA_real_
            min_s <- NA_real_
            max_s <- NA_real_
            result_rows <- NA_integer_

            if (!is.null(creation_errors[[spec$base_name]])) {
                status <- "unsupported"
                err <- creation_errors[[spec$base_name]]
                runtime_name <- ""
            } else {
                benchmark <- NULL
                tryCatch({
                    benchmark <- bench_callable(
                        function() run_case(
                            operation,
                            runtime_name,
                            spec$threshold,
                            interval_cache[[profile$case_suffix]],
                            profile$iterator
                        ),
                        opts$warmup,
                        opts$reps
                    )
                }, error = function(e) {
                    status <<- "error"
                    err <<- sprintf("%s: %s", class(e)[1], conditionMessage(e))
                })

                if (!is.null(benchmark)) {
                    median_s <- benchmark$median_s
                    std_s <- benchmark$std_s
                    min_s <- benchmark$min_s
                    max_s <- benchmark$max_s
                    result_rows <- benchmark$result_rows
                }
            }

            append_row(
                impl = "rmisha",
                case_id = case_id,
                operation = operation,
                vtrack_label = spec$base_name,
                vtrack_name = runtime_name,
                vtrack_func = spec$func,
                source_track = ifelse(is.null(spec$src), "NULL", spec$src),
                source_density = spec$source_density,
                profile = profile$case_suffix,
                chrom_mode = profile$chrom_mode,
                size_label = profile$size_label,
                iterator = profile$iterator,
                iterator_density = profile$iterator_density,
                warmup = opts$warmup,
                reps = opts$reps,
                status = status,
                median_s = median_s,
                std_s = std_s,
                min_s = min_s,
                max_s = max_s,
                result_rows = result_rows,
                error = err,
                timestamp_utc = timestamp_utc,
                package_path = opts$rmisha_src
            )

            if (!opts$quiet) {
                cat(sprintf("[%03d/%03d] %s: %s\n", case_idx, total_cases, case_id, status))
            }
        }
    }
}

if (length(rows) == 0L) {
    final_df <- data.frame(stringsAsFactors = FALSE)
} else {
    final_df <- do.call(rbind, rows)
}

if (!is.null(opts$output_csv)) {
    output_csv <- normalizePath(path.expand(opts$output_csv), winslash = "/", mustWork = FALSE)
    out_dir <- dirname(output_csv)
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    }
    write.csv(final_df, output_csv, row.names = FALSE, quote = TRUE)
}

if (!opts$quiet) {
    success_n <- sum(final_df$status == "success")
    unsupported_n <- sum(final_df$status == "unsupported")
    error_n <- sum(final_df$status == "error")
    cat(sprintf(
        "Completed R misha benchmark suite: total=%d success=%d unsupported=%d errors=%d\n",
        nrow(final_df), success_n, unsupported_n, error_n
    ))
    cat(sprintf("R misha source path: %s\n", opts$rmisha_src))
    cat(sprintf("DB root: %s\n", opts$db_root))
}
