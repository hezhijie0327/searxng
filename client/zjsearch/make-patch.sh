#!/usr/bin/env bash
# Export the zjsearch theme changes (client sources + server data templates)
# as a single patch to apply on another machine:
#
#   ./make-patch.sh [base-ref] [out-file]
#   # on the target machine:
#   git apply --check zjsearch-theme.patch && git apply zjsearch-theme.patch
#   cd client/zjsearch && npm install && npm run build
#
# Default base: the fork's master branch (falls back to origin/master, then
# d834a52), so the patch carries the whole theme delta relative to master.
# Only client/zjsearch and searx/templates/zjsearch are included; both are
# required - the map-page fix lives in the data templates, not the client.

set -euo pipefail

base=${1:-}
if [ -z "$base" ]; then
  for ref in master origin/master origin/main d834a52; do
    if git rev-parse -q --verify "$ref" >/dev/null 2>&1; then
      base=$ref
      break
    fi
  done
fi
out=${2:-zjsearch-theme.patch}

cd "$(dirname "$0")/../.."

git diff --text "$base" HEAD -- client/zjsearch searx/templates/zjsearch > "$out"

files=$(grep -c '^diff --git' "$out" || true)
if [ "$files" = "0" ]; then
  echo "no changes between $base and HEAD - nothing to patch" >&2
  exit 1
fi
echo "wrote $out: $files files, $base..$(git rev-parse --short HEAD)"
grep '^diff --git' "$out" | sed 's/^diff --git a\///; s/ b\/.*//; s/^/  /'
