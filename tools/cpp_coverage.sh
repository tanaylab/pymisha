#!/usr/bin/env bash
# Build pymisha with gcov instrumentation, run tests, and report C++ line coverage.
#
# Usage:
#   bash tools/cpp_coverage.sh              # terminal summary
#   bash tools/cpp_coverage.sh --html       # also generate lcov HTML report in cpp-coverage/
#
# Restores the optimized (non-instrumented) build when done.
# Requires: gcov. For --html: lcov (apt install lcov).
set -euo pipefail

cd "$(dirname "$0")/.."

HTML=0
[[ "${1:-}" == "--html" ]] && HTML=1

echo "==> Cleaning previous build and coverage data..."
/bin/rm -rf build/ _pymisha*.so cpp-coverage/ cpp-coverage*.info 2>/dev/null || true

echo "==> Building with gcov instrumentation..."
PYMISHA_COVERAGE=1 python setup.py build_ext --inplace 2>&1 | tail -3

echo "==> Running test suite..."
python -m pytest tests/ -x -q --override-ini="addopts=" --tb=line 2>&1 | tail -5

# .gcda files land next to .o files in build/
GCDA_DIR=$(find build/ -name '*.gcda' -printf '%h\n' 2>/dev/null | sort -u | head -1)
if [[ -z "$GCDA_DIR" ]]; then
    echo "ERROR: no .gcda files found. Coverage instrumentation was not applied."
    exit 1
fi
echo "==> Found gcov data in $GCDA_DIR"

if [[ "$HTML" -eq 1 ]]; then
    command -v lcov >/dev/null 2>&1 || { echo "lcov not found (apt install lcov)"; exit 1; }
    echo "==> Generating lcov HTML report..."
    lcov --capture --directory "$GCDA_DIR" -o cpp-coverage-raw.info --gcov-tool gcov -q
    lcov --remove cpp-coverage-raw.info '/usr/*' '*/numpy/*' '*/python*' \
        -o cpp-coverage.info -q
    genhtml cpp-coverage.info -o cpp-coverage -q
    echo "==> HTML report: cpp-coverage/index.html"
    lcov --summary cpp-coverage.info 2>&1 | tail -3
else
    echo ""
    echo "==> C++ Coverage Summary"
    echo "========================"
    (cd "$GCDA_DIR" && gcov *.gcda 2>/dev/null) | \
        grep -B1 '^Lines executed:' | \
        grep -A1 "src/" | \
        grep -v '^--$' | \
        sed "s|^File '.*src/|  |; s|'$||" | \
        paste - - | column -t -s$'\t' | \
        sort -t: -k2 -rn
fi

echo ""
echo "==> Restoring optimized build..."
/bin/rm -rf build/ _pymisha*.so
python setup.py build_ext --inplace 2>&1 | tail -3
echo "==> Done."
