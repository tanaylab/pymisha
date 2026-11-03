#ifndef _GINTERVALS_H_INCLUDED_
#define _GINTERVALS_H_INCLUDED_

#include <vector>
#include "GInterval.h"

// pymisha has none of misha's GIntervalsFetcher1D machinery. BigWig.h, shared verbatim with misha,
// only fills a GIntervals as a vector, so that is all this is.
typedef std::vector<GInterval> GIntervals;

#endif
