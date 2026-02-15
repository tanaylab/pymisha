# PyMisha Development Guide

## Branch Model

| Branch | Purpose | Remote |
|--------|---------|--------|
| `dev` | Active development. All work happens here. | `private` (aviezerl/pymisha) |
| `main` | Clean release history. One squash commit per release. | `origin` (tanaylab/pymisha) |

### Rules

- **All commits go to `dev` first.** Never commit directly to `main`.
- **`main` gets squash-merged releases only.** Each release is a single commit on `main` with a clean summary message.
- **Tags live on `main`.** Version tags (`v0.1.0`, `v0.1.1`, ...) are created on `main` after the squash commit.

## Pushing to main (release workflow)

1. **Ensure dev is clean and tested:**
   ```bash
   git checkout dev
   python -m pytest tests/ -x -q   # all tests must pass
   ```

2. **Bump version** in `pyproject.toml` and update `CHANGELOG.md`:
   - Patch (`0.1.x`): bug fixes, performance, docs
   - Minor (`0.x.0`): new features, API additions
   - Major (`x.0.0`): breaking API changes

3. **Commit the version bump on dev:**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: vX.Y.Z — short description"
   ```

4. **Squash-merge dev onto main:**
   ```bash
   git checkout main
   git read-tree --reset -u dev          # replace main's tree with dev's content
   git commit -m "vX.Y.Z: summary of changes"
   ```

5. **Tag and push:**
   ```bash
   git tag vX.Y.Z
   git push origin main
   git push origin vX.Y.Z                # triggers PyPI publish via GitHub Actions
   ```

6. **Push dev to private:**
   ```bash
   git checkout dev
   git push private dev
   ```

The `v*` tag triggers `.github/workflows/publish.yml` which builds wheels (Linux x86_64, macOS x86_64/arm64, Python 3.10-3.12) and publishes to PyPI via trusted publishing.

## Development workflow

- **Rebuild after C++ changes:** `pip install -e .`
- **Run tests:** `pytest tests/ -x -q`
- **Run benchmarks:** `python tests/bench_cpp_perf.py --json`
- **Build docs locally:** `mkdocs serve`

## Project structure

```
pymisha/           Python package (public API)
src/               C++ extension sources
tests/             pytest test suite + benchmarks
docs/              MkDocs Material documentation site
dev/               Development skills, notes, internal docs
.github/workflows/ CI (tests), publish (PyPI), docs (GitHub Pages)
```

## Key conventions

- C++ streaming functions: parse args -> init scanner -> stream -> collect -> build DataFrame
- Python routing: check `_find_vtracks_in_expr()` -> C++ if clean, Python fallback for vtracks
- Test DB at `tests/testdb/trackdb/test` with chroms `1` (500k), `2` (300k), `X` (200k)
- Tests that create tracks must call `_pymisha.pm_dbreload()` after cleanup
