// pymisha does not link R. Parquet.cpp is kept identical to R misha's copy (R misha's
// tools/check-shared-cpp.py compares the C++ files the two packages share), and it includes
// "rdbutils.h" for rdb::check_interrupt (and, through it, <stdexcept>). This header stands in
// for R misha's: src/shims is on pymisha's include path, and src has no rdbutils.h of its own,
// so that the drift check does not take this file for a copy of R misha's.

#ifndef PYMISHA_SHIMS_RDBUTILS_H_
#define PYMISHA_SHIMS_RDBUTILS_H_

#include <stdexcept>

#include "pymisha.h"

namespace rdb {

inline void check_interrupt() { ::check_interrupt(); }

}

#endif
