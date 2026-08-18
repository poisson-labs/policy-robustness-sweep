#!/usr/bin/env bash
# Fetch the pinned Rerun web viewer from npm and verify against the committed hash.
# Usage: scripts/fetch_viewer.sh <dest_dir>
# Bytes never live in git; the pin (VERSION) and hash (SHA256SUMS) do.
set -euo pipefail
DEST="${1:?dest dir}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$HERE/web/vendor/rerun/VERSION")"
EXPECTED="$(awk '{print $1}' "$HERE/web/vendor/rerun/SHA256SUMS" | head -1)"
URL="https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-${VERSION}.tgz"
TMP="$(mktemp -d)"
curl -sSL "$URL" -o "$TMP/wv.tgz"
ACTUAL="$(shasum -a 256 "$TMP/wv.tgz" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "sha256 mismatch: expected $EXPECTED got $ACTUAL" >&2; exit 1
fi
tar -xzf "$TMP/wv.tgz" -C "$TMP"
mkdir -p "$DEST"
cp "$TMP/package/re_viewer.js" "$TMP/package/re_viewer_bg.wasm" "$TMP/package/index.js" "$DEST/"
# The vendored index.js does a bare `import("./re_viewer")` (bundler idiom); static
# hosts serve extensionless files with no/incorrect Content-Type, which breaks module
# loading. Rewrite that one specifier to the real filename (verified byte-identical
# otherwise). Header overrides on the bare path proved unreliable under `netlify dev`.
sed -i.bak 's#import("./re_viewer")#import("./re_viewer.js")#g' "$DEST/index.js" && rm -f "$DEST/index.js.bak"
grep -q 'import("./re_viewer.js")' "$DEST/index.js" || { echo "specifier rewrite failed" >&2; exit 1; }
echo "$VERSION" > "$DEST/VERSION"
echo "viewer $VERSION -> $DEST (sha256 verified)"
