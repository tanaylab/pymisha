// pymisha does not link R. StatQuadTreeCachedSerializer.h is kept identical to R
// misha's copy (R misha's tools/check-shared-cpp.py compares the C++ files the two
// packages share), and it includes <R_ext/Print.h> for REprintf. This header, found
// first on pymisha's include path (src/), stands in for R's: REprintf writes to stderr.

#ifndef PYMISHA_R_EXT_PRINT_H_
#define PYMISHA_R_EXT_PRINT_H_

#include <cstdarg>
#include <cstdio>

inline void REprintf(const char *format, ...)
{
	va_list ap;
	va_start(ap, format);
	vfprintf(stderr, format, ap);
	va_end(ap);
}

#endif
