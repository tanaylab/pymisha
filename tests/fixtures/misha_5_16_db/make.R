# Builds the big interval sets of tests/test_misha_5_16_db.py with R misha 5.16.0, as its
# test-review-2026-11-dbstate.R makes them (pymisha writes no big sets):
#   u3.interv   80 rows on chr1, chr2, chrX, chrM, converted to indexed (remove.old = TRUE)
#   u3b.interv  u3, then chrX updated to 2 rows (its own file beside the index)
#   u2d.interv  a 2D set of 20 rows on chr1-chr1 and chr2-chrX, converted to indexed
#   at1.interv  10 rows on chr1 and 30 on chrX, per-chromosome, with the attribute source = "lab"
#   bs1.interv  chr1 and chr2 0-10, converted to indexed
#   bs2.interv  chr1 0-10 and 40-50, chr2 0-10, converted to indexed (bs1 re-created)
#   dash.interv 15 rows on chr1, HLA-A and scaf-2, per-chromosome (a per-chromosome database)
#
#   Rscript make.R <out_dir> [misha_lib]
args <- commandArgs(TRUE)
out <- normalizePath(args[1])
if (length(args) > 1) library(misha, lib.loc = args[2]) else library(misha)
options(gmultitasking = FALSE)

td <- tempfile()
dir.create(td)
fa <- file.path(td, "db.fa")
chroms <- c("chr1", "chr2", "chrX", "chrM")
set.seed(17)
seqs <- vapply(chroms, function(x) paste(sample(c("A", "C", "G", "T"), 5000, TRUE), collapse = ""), "")
writeLines(paste0(">", chroms, "\n", seqs), fa)
db <- file.path(td, "db")
gdb.create(groot = db, fasta = fa, format = "indexed", verbose = FALSE)
gsetroot(db)

d <- data.frame(chrom = rep(c("chr1", "chr2", "chrX", "chrM"), each = 20), start = rep(seq(0, 1900, 100), 4))
d$end <- d$start + 50
withr::with_options(list(gmax.data.size = 50), gintervals.save("u3", d))
gintervals.convert_to_indexed("u3", remove.old = TRUE)
withr::with_options(list(gmax.data.size = 50), gintervals.save("u3b", d))
gintervals.convert_to_indexed("u3b", remove.old = TRUE)
withr::with_options(list(gmax.data.size = 50), gintervals.update("u3b", gintervals("chrX", c(0, 500), c(10, 510)), chrom = "chrX"))

d2 <- data.frame(chrom1 = rep(c("chr1", "chr2"), each = 10), start1 = rep(seq(0, 900, 100), 2), chrom2 = rep(c("chr1", "chrX"), each = 10), start2 = 0)
d2$end1 <- d2$start1 + 50
d2$end2 <- d2$start2 + 50
d2 <- d2[, c("chrom1", "start1", "end1", "chrom2", "start2", "end2")]
withr::with_options(list(gmax.data.size = 15), gintervals.save("u2d", d2))
gintervals.2d.convert_to_indexed("u2d", remove.old = TRUE)

withr::with_options(list(gmax.data.size = 30, gmulticontig.indexed_format = FALSE), {
    gintervals.save("at1", gintervals("chr1", seq(0, 900, 100), seq(50, 950, 100)))
    gintervals.attr.set("at1", "source", "lab")
    gintervals.update("at1", gintervals("chrX", seq(0, 2900, 100), seq(50, 2950, 100)), chrom = "chrX")
})
stopifnot(dir.exists(file.path(db, "tracks", "at1.interv")), gintervals.attr.get("at1", "source") == "lab")

withr::with_options(list(gmax.data.size = 1), {
    gintervals.save("bs1", gintervals(c("chr1", "chr2"), 0, 10))
    gintervals.save("bs2", gintervals(c("chr1", "chr1", "chr2"), c(0, 40, 0), c(10, 50, 10)))
})
gintervals.convert_to_indexed("bs1", remove.old = TRUE)
gintervals.convert_to_indexed("bs2", remove.old = TRUE)

# a per-chromosome database with contigs whose names have a dash
pdb <- file.path(td, "pdb")
sizes <- c(chr1 = 10000, "HLA-A" = 5000, "scaf-2" = 5000)
pfa <- file.path(td, paste0(names(sizes), ".fa"))
for (i in seq_along(sizes)) writeLines(c(paste0(">", names(sizes)[i]), strrep("A", sizes[i])), pfa[i])
gdb.create(groot = pdb, fasta = pfa, format = "per-chromosome", verbose = FALSE)
gsetroot(pdb)
dd <- data.frame(chrom = rep(c("chr1", "HLA-A", "scaf-2"), each = 5), start = rep(seq(0, 400, 100), 3))
dd$end <- dd$start + 50
withr::with_options(list(gmax.data.size = 4, gmulticontig.indexed_format = FALSE), gintervals.save("dash", dd))
stopifnot(file.exists(file.path(pdb, "tracks", "dash.interv", "HLA-A")))

for (s in c("u3", "u3b", "u2d", "at1", "bs1", "bs2")) {
    file.copy(file.path(db, "tracks", paste0(s, ".interv")), out, recursive = TRUE)
}
file.copy(file.path(pdb, "tracks", "dash.interv"), out, recursive = TRUE)
for (s in c("u3", "u3b", "u2d", "at1", "bs1", "bs2", "dash")) cat(s, ":", list.files(file.path(out, paste0(s, ".interv")), all.files = TRUE, no.. = TRUE), "\n")
