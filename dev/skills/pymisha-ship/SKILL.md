# Ship: dev → main

Ship changes from the `dev` branch to a clean `main` branch via squash merge, excluding development-only files.

## Remote Layout

| Remote | Repository | Branch | Purpose |
|--------|-----------|--------|---------|
| `private` | `aviezerl/pymisha` | `dev` | Development (messy history, dev artifacts) |
| `origin` | `tanaylab/pymisha` | `main` | Public release (clean commits, no dev files) |

**Guardrails:**
- `dev` must NEVER be pushed to `origin` (tanaylab)
- `main` must NEVER be pushed to `private` (aviezerl)
- The ship script enforces this automatically

## Dev-Only Files (excluded from main)

These paths exist on `dev` but are removed during shipping:

- `dev/` — implementation notes, skills, planning docs
- `CLAUDE.md` — agent instruction file (symlink to deleted AGENTS.md)

## Usage

### Via script

```bash
# Review what will ship (dry run)
bash dev/skills/pymisha-ship/ship.sh

# Ship with a commit message
bash dev/skills/pymisha-ship/ship.sh "Add parallel gsynth and per-chromosome DB format"

# Ship and push to tanaylab/pymisha
bash dev/skills/pymisha-ship/ship.sh "Add parallel gsynth" --push
```

### Via Claude Code

Ask Claude to `/ship` or say "ship to main" and it will:
1. Run `dev/skills/pymisha-ship/ship.sh`
2. Show you the diff summary
3. Ask for your commit message
4. Optionally push to origin

### Manual steps (what the script does)

```bash
git checkout main
git merge --squash dev
git rm -rf dev/
git rm -f CLAUDE.md 2>/dev/null
git add -A
git commit -m "your message"
git push origin main        # tanaylab/pymisha
git checkout dev
```

## Rollback

If something goes wrong mid-ship:

```bash
git checkout -f dev          # go back to dev, discard any staging mess
git branch -D main           # delete broken main
git checkout -b main origin/main  # restore from remote
```

## First-Time Setup

If `main` branch doesn't exist yet:

```bash
git checkout --orphan main
git rm -rf --cached dev/ CLAUDE.md
git commit -m "initial commit"
git checkout -f dev
git remote add origin git@github.com:tanaylab/pymisha.git
```
