# Regenerate tests/r_parity/scope_parity_baseline.json from R misha.
#
# The scope-canonicalisation parity test compares pymisha against FROZEN R
# output rather than a live R process, so it runs in ordinary CI with no R
# installed - the same trade tests/r_parity/baseline.py makes for the .rds
# snapshots. Re-run this only when the cases below change, or to re-confirm
# against a newer misha:
#
#   Rscript tools/generate_scope_parity_baseline.R > tests/r_parity/scope_parity_baseline.json
#
# Everything runs on misha's own bundled example database, so it reproduces
# anywhere misha is installed.

suppressMessages(library(misha)); suppressMessages(library(jsonlite))
gdb.init_examples()
options(gmultitasking = FALSE)

SC <- list(
  single       = data.frame(chrom="chr1", start=0,             end=10000),
  overlapping  = data.frame(chrom="chr1", start=c(0,5000),     end=c(10000,15000)),
  nested       = data.frame(chrom="chr1", start=c(0,2000),     end=c(10000,4000)),
  touching     = data.frame(chrom="chr1", start=c(0,10000),    end=c(10000,20000)),
  unsorted     = data.frame(chrom="chr1", start=c(5000,0),     end=c(15000,10000)),
  disjoint     = data.frame(chrom="chr1", start=c(0,20000),    end=c(10000,30000)),
  multichrom   = data.frame(chrom=c("chr1","chr2","chr1"), start=c(0,0,5000), end=c(10000,10000,15000)),
  dup          = data.frame(chrom="chr1", start=c(0,0),        end=c(10000,10000))
)

# Iterator variants. The DataFrame iterator matters as much as the scope shape:
# pymisha intersects it with the scope in Python and hands the resulting BINS to
# C++, so anything that canonicalises there merges adjacent bins and silently
# changes the answer. A whole class of divergence was invisible until this axis
# was added - gdist with a two-bin touching iterator returned 1 bin where misha
# returned 2, and had done so before any of this work.
ITR <- list(
  auto  = NULL,
  fixed = 500,
  df    = data.frame(chrom=c("chr1","chr1"), start=c(0,1000), end=c(1000,2000))
)

out <- list()
for (nm in names(SC)) {
 for (inm in names(ITR)) {
  iv <- SC[[nm]]
  it <- ITR[[inm]]
  key <- paste0(nm, "|", inm)
  g <- function(f) tryCatch(f(), error=function(e) paste("ERROR:", conditionMessage(e)))
  out[[key]] <- list(
    gsummary   = g(function() as.numeric(gsummary("dense_track", intervals=iv, iterator=it))),
    gquantiles = g(function() as.numeric(gquantiles("dense_track", c(0.1,0.5,0.9), intervals=iv, iterator=it))),
    gcor       = g(function() as.numeric(unlist(gcor("dense_track","dense_track*2", intervals=iv, iterator=it)))),
    gscreen_n  = g(function() { r <- gscreen("dense_track > 0.1", intervals=iv, iterator=it); if (is.null(r)) 0 else nrow(r) }),
    gscreen    = g(function() { r <- gscreen("dense_track > 0.1", intervals=iv, iterator=it); if (is.null(r)) NULL else as.numeric(c(r$start, r$end)) }),
    gextract_n = g(function() { r <- gextract("dense_track", intervals=iv, iterator=it); if (is.null(r)) 0 else nrow(r) }),
    gdist      = g(function() as.numeric(gdist("dense_track", seq(0,1,by=0.25), intervals=iv, iterator=it))),
    gsegment_n = g(function() { r <- gsegment("dense_track", 500, 0.5, intervals=iv); if (is.null(r)) 0 else nrow(r) }),
    gpartition_n = g(function() { r <- gpartition("dense_track", seq(0,1,by=0.25), intervals=iv, iterator=it); if (is.null(r)) 0 else nrow(r) })
  )
 }
}
cat(toJSON(out, digits=10, auto_unbox=TRUE, null="null", na="string"))
