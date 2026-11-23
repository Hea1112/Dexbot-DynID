#!/usr/bin/env bash
# Create the public Dexbot-DynID GitHub repository from this reviewed source tree.
# No repository is created until the operator approves the interactive prompt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v git >/dev/null 2>&1; then
  echo 'git is not installed. Install git and try again.' >&2
  exit 2
fi
if ! command -v gh >/dev/null 2>&1; then
  echo 'GitHub CLI (gh) is not installed. Install gh or follow docs/GITHUB_PUBLISH.md.' >&2
  exit 2
fi
if ! gh auth status >/dev/null 2>&1; then
  echo 'GitHub CLI is not authenticated. Run: gh auth login' >&2
  exit 2
fi
if ! python3 -m pytest -q; then
  echo 'Tests did not pass; publication aborted.' >&2
  exit 2
fi
if [[ ! -d .git ]]; then
  git init -b main
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo 'A remote origin already exists; inspect it before publishing again.' >&2
  exit 2
fi

# Explicit source allowlist: never stage trajectory CSVs, NPZ matrices, URDFs or CAD.
git add -- \
  .github/workflows/ci.yml .gitignore LICENSE README.md README_zh.md \
  pyproject.toml requirements.txt requirements-sdp.txt \
  configs/manuscript_protocol.json data/INPUT_FORMAT.md \
  docs/*.md docs/SOURCE_SHA256_MANIFEST.txt \
  scripts/*.py scripts/publish_github.sh \
  src/dexbot_id/*.py tests/*.py

git diff --cached --check
echo 'Files to be committed:'
git diff --cached --name-only
printf '\nPublish these files as a PUBLIC GitHub repository named Dexbot-DynID? [yes/NO] '
read -r answer
if [[ "$answer" != yes ]]; then
  echo 'Publication canceled. No GitHub repository was created.'
  exit 0
fi

if ! git diff --cached --quiet; then
  git commit -m 'Initial release of Dexbot-DynID'
fi
# GitHub CLI creates the repository under the authenticated account.
gh repo create Dexbot-DynID --public --source=. --remote=origin --push
gh repo view --json url --jq .url
