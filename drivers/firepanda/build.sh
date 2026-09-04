#!/usr/bin/env bash
# Builds the driver and stamps what it was built from.
#
# The stamp is the point of this script existing rather than a line in a README. A
# result file that says firepanda scored 84 percent and does not say which firepanda
# is not a result, and the driver cannot answer that question itself: the library has
# no version constant in Mojo, so the only place the answer exists at build time is
# the checkout it was compiled against. Stamping it here means the version travels
# with the binary and is regenerated every build, which is the one arrangement that
# cannot go stale.
#
# The Mojo toolchain is pinned by the firepanda repository and not by this one. Two
# pins for one toolchain is how a driver ends up compiled against a library it does
# not match, so this borrows theirs.
#
# Usage:
#     drivers/firepanda/build.sh [path to a firepanda checkout]
#
# The default is a sibling directory, which is how the three repositories sit on a
# development machine and in the CI job that builds this.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="$here/main.mojo"
binary="$here/firepanda-compat-driver"
stamp="$here/stamp.json"

firepanda="${1:-$(cd "$here/../../.." && pwd)/firepanda}"
if [ ! -d "$firepanda/firepanda" ]; then
  echo "no firepanda checkout at $firepanda" >&2
  echo "usage: $0 [path to a firepanda checkout]" >&2
  exit 2
fi

cd "$firepanda"
pixi run mojo build -I . "$source_file" -o "$binary"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$firepanda/pixi.toml" | head -1)
commit=$(git -C "$firepanda" rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git -C "$firepanda" status --porcelain 2>/dev/null | head -1)
mojo=$(pixi run mojo --version 2>/dev/null | head -1)

cat > "$stamp" <<JSON
{
  "firepanda": "${version:-unknown}",
  "commit": "${commit}${dirty:+ (dirty)}",
  "mojo": "${mojo:-unknown}",
  "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "built $binary"
cat "$stamp"
