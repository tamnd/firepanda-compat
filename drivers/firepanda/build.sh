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
#     drivers/firepanda/build.sh [library checkout] [toolchain checkout]
#
# The default for the library is a sibling directory, which is how the three
# repositories sit on a development machine and in the CI job that builds this. The
# toolchain defaults to the same place and almost always stays there.
#
# The two are separable for one workflow, which is the whole point of this repository
# existing: find a conformance failure, fix it on a firepanda branch, rebuild the
# driver and see whether the number moved. A git worktree is the sane way to hold that
# branch, because the main checkout usually has somebody else's work in progress in
# it, and a worktree has no pixi environment of its own. Installing a second one to
# compile a branch of the same library against the same pinned toolchain is minutes of
# downloading for nothing. So point the library at the worktree and let the toolchain
# stay where it is.
#
# The stamp still reads the version, the commit and the dirty flag from the library
# checkout, which is the one that decides what was measured. Only `mojo` itself comes
# from the toolchain.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="$here/main.mojo"
binary="$here/firepanda-compat-driver"
stamp="$here/stamp.json"

firepanda="${1:-$(cd "$here/../../.." && pwd)/firepanda}"
if [ ! -d "$firepanda/firepanda" ]; then
  echo "no firepanda checkout at $firepanda" >&2
  echo "usage: $0 [library checkout] [toolchain checkout]" >&2
  exit 2
fi
firepanda="$(cd "$firepanda" && pwd)"

toolchain="${2:-$firepanda}"
if [ ! -f "$toolchain/pixi.toml" ]; then
  echo "no pixi.toml at $toolchain, so there is no toolchain to build with" >&2
  echo "usage: $0 [library checkout] [toolchain checkout]" >&2
  exit 2
fi
toolchain="$(cd "$toolchain" && pwd)"

# Built from the toolchain checkout with the library named by an absolute path,
# rather than from the library with `-I .`, so that the two can be different
# directories. When they are the same directory this is the same command.
cd "$toolchain"
pixi run mojo build -I "$firepanda" "$source_file" -o "$binary"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$firepanda/pixi.toml" | head -1)
commit=$(git -C "$firepanda" rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git -C "$firepanda" status --porcelain 2>/dev/null | head -1)
mojo=$(pixi run mojo --version 2>/dev/null | head -1)
toolchain_note=""
[ "$toolchain" != "$firepanda" ] && toolchain_note=" (toolchain from $toolchain)"

cat > "$stamp" <<JSON
{
  "firepanda": "${version:-unknown}",
  "commit": "${commit}${dirty:+ (dirty)}",
  "mojo": "${mojo:-unknown}${toolchain_note}",
  "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "built $binary"
cat "$stamp"
