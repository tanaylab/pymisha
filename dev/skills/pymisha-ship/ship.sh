#!/usr/bin/env bash
# Ship dev → main: squash merge dev into main, excluding dev-only files.
#
# Usage:
#   ship.sh                       # dry run: show what would ship
#   ship.sh "commit message"      # ship with message
#   ship.sh "commit message" --push  # ship and push to origin (tanaylab)
#
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ── Config ──────────────────────────────────────────────────────────────
DEV_REMOTE="private"        # aviezerl/pymisha
MAIN_REMOTE="origin"        # tanaylab/pymisha
DEV_BRANCH="dev"
MAIN_BRANCH="main"
DEV_ONLY_PATHS=("dev/" "CLAUDE.md" "AGENTS.md")  # removed from main after merge

# ── Helpers ─────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

cleanup() {
    # Always return to dev on exit
    local current
    current="$(git branch --show-current 2>/dev/null || true)"
    if [[ "$current" != "$DEV_BRANCH" ]]; then
        info "Returning to $DEV_BRANCH branch..."
        git checkout -f "$DEV_BRANCH" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Guardrails ──────────────────────────────────────────────────────────
guard_remote() {
    local branch="$1" remote="$2" action="$3"
    if [[ "$branch" == "$DEV_BRANCH" && "$remote" == "$MAIN_REMOTE" ]]; then
        die "BLOCKED: Cannot $action $DEV_BRANCH to $MAIN_REMOTE (tanaylab). Use $DEV_REMOTE instead."
    fi
    if [[ "$branch" == "$MAIN_BRANCH" && "$remote" == "$DEV_REMOTE" ]]; then
        die "BLOCKED: Cannot $action $MAIN_BRANCH to $DEV_REMOTE (aviezerl). Use $MAIN_REMOTE instead."
    fi
}

# ── Pre-checks ──────────────────────────────────────────────────────────
[[ "$(git branch --show-current)" == "$DEV_BRANCH" ]] \
    || die "Must be on $DEV_BRANCH branch. Currently on: $(git branch --show-current)"

[[ -z "$(git status --porcelain)" ]] \
    || die "Working tree is not clean. Commit or stash changes first."

git rev-parse --verify "$MAIN_BRANCH" >/dev/null 2>&1 \
    || die "Branch '$MAIN_BRANCH' does not exist. Create it first (see SKILL.md)."

# ── Parse args ──────────────────────────────────────────────────────────
COMMIT_MSG="${1:-}"
DO_PUSH=false
for arg in "$@"; do
    [[ "$arg" == "--push" ]] && DO_PUSH=true
done

# ── Ship ────────────────────────────────────────────────────────────────
info "Switching to $MAIN_BRANCH..."
git checkout "$MAIN_BRANCH"

info "Replacing $MAIN_BRANCH tree with $DEV_BRANCH content..."
git read-tree --reset -u "$DEV_BRANCH"

# Remove dev-only files
for path in "${DEV_ONLY_PATHS[@]}"; do
    if git ls-files --error-unmatch "$path" >/dev/null 2>&1 || [[ -e "$path" ]]; then
        info "Removing dev-only: $path"
        git rm -rf "$path" 2>/dev/null || rm -rf "$path"
    fi
done

# ── Show summary ────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Ship Summary (dev → main)"
echo "══════════════════════════════════════════════"
git diff --cached --stat
echo ""

if [[ -z "$COMMIT_MSG" ]]; then
    info "DRY RUN: No commit message provided."
    echo ""
    echo "To commit:  git commit -m \"your message\""
    echo "To push:    git push $MAIN_REMOTE $MAIN_BRANCH"
    echo "To abort:   git checkout -f $DEV_BRANCH"
    echo ""
    # Don't return to dev — user wants to review and commit manually
    trap - EXIT
    exit 0
fi

# ── Commit ──────────────────────────────────────────────────────────────
info "Committing: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

if $DO_PUSH; then
    guard_remote "$MAIN_BRANCH" "$MAIN_REMOTE" "push"
    info "Pushing $MAIN_BRANCH to $MAIN_REMOTE..."
    git push "$MAIN_REMOTE" "$MAIN_BRANCH"

    # ── Tag ────────────────────────────────────────────────────────────
    # Extract version tag from pyproject.toml (format: version = "X.Y.Z")
    VERSION="$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')"
    TAG="v${VERSION}"

    if git rev-parse "$TAG" >/dev/null 2>&1; then
        info "Tag $TAG already exists, skipping."
    else
        info "Tagging $TAG..."
        git tag "$TAG"
        git push "$MAIN_REMOTE" "$TAG"
    fi

    # ── GitHub release ─────────────────────────────────────────────────
    if command -v gh >/dev/null 2>&1; then
        if gh release view "$TAG" >/dev/null 2>&1; then
            info "GitHub release $TAG already exists, skipping."
        else
            # Extract release notes from CHANGELOG.md: everything between
            # the first ## vX.Y.Z heading and the next ## heading.
            NOTES="$(awk '/^## v'"${VERSION}"'/{found=1; next} /^## /{if(found) exit} found{print}' CHANGELOG.md)"
            if [[ -n "$NOTES" ]]; then
                info "Creating GitHub release $TAG..."
                gh release create "$TAG" --title "$TAG" --notes "$NOTES"
            else
                info "Creating GitHub release $TAG (using commit message)..."
                gh release create "$TAG" --title "$TAG" --notes "$COMMIT_MSG"
            fi
        fi
    else
        info "gh CLI not found — skipping GitHub release creation."
        info "Create manually: gh release create $TAG --title \"$TAG\" --notes \"...\""
    fi

    # ── Push dev to private ────────────────────────────────────────────
    info "Returning to $DEV_BRANCH to push to $DEV_REMOTE..."
    git checkout "$DEV_BRANCH"
    guard_remote "$DEV_BRANCH" "$DEV_REMOTE" "push"
    git push "$DEV_REMOTE" "$DEV_BRANCH"
fi

info "Done! Returning to $DEV_BRANCH."
# cleanup trap handles returning to dev
