/*
 * PMTrackExpression2DIterator.cpp
 */

#include "PMTrackExpression2DIterator.h"

PMTrackExpressionIntervals2DIterator::PMTrackExpressionIntervals2DIterator(
    const std::vector<GInterval2D> &intervals)
    : m_intervals(intervals),
      m_cur_idx(0),
      m_isend(intervals.empty())
{
}

void PMTrackExpressionIntervals2DIterator::begin()
{
    m_cur_idx = 0;
    m_isend = m_intervals.empty();
}

void PMTrackExpressionIntervals2DIterator::next()
{
    if (m_isend) return;
    ++m_cur_idx;
    if (m_cur_idx >= m_intervals.size()) {
        m_isend = true;
    }
}
