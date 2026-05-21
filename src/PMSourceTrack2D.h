#ifndef PMSOURCETRACK2D_H
#define PMSOURCETRACK2D_H

#include <cstdint>
#include <string>
#include <vector>

// Rectangles from a 2D source-track directory. Per-pair {c1}-{c2} files
// (R-style "chrA-chrB" or pymisha-style "1-2") are scanned and their
// quadtree objects enumerated. POINTS objects are unified into RECTS shape
// via (x, x+1, y, y+1, value).
//
// chrom1[i]/chrom2[i] are taken verbatim from the file name (R uses
// the source assembly's raw chrom names; chain matching downstream is
// against `chromsrc`).
//
// Same directory cannot mix RECTS and POINTS files - that throws
// std::invalid_argument.
struct SourceTrack2DRows {
    std::vector<std::string> chrom1;
    std::vector<std::string> chrom2;
    std::vector<int64_t>     x1;
    std::vector<int64_t>     y1;
    std::vector<int64_t>     x2;
    std::vector<int64_t>     y2;
    std::vector<double>      value;
    bool                     is_points = false;
};

// Read a 2D source-track directory. APPENDS rows to out (caller pre-clears).
// out.is_points is set once a per-pair file is encountered.
//
// Per-pair file naming: anything not prefixed by '.' and not equal to
// "track.idx"/"track.dat" is treated as a per-pair file. The file's signature
// (int32 at offset 0) is checked against -9 (RECTS) / -10 (POINTS); any other
// signature is ignored (lets 1D sources sharing a dir not break the read).
//
// Throws std::invalid_argument on data-format errors, std::runtime_error
// on I/O errors.
void read_source_track_2d_cpp(
    const std::string &src_track_dir,
    SourceTrack2DRows &out);

#endif  // PMSOURCETRACK2D_H
