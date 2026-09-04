"""The `str` accessor.

Fifty odd methods and almost all of them are one line of intent with a page of edge
cases underneath, which is why this section is the largest one. The four string frames
are doing different jobs: ascii is the boring baseline, unicode is where a length in
characters stops being a length in bytes, null heavy is where every method has to
decide what a null maps to, and pattern is the one built to be matched against.

The null answer is the same everywhere and it is worth stating once. A string method
on a null gives a null, not an empty string and not a false, and the predicates are
the ones where that is easiest to get wrong because a false looks so reasonable.
"""

from __future__ import annotations

from fpcompat.cases import case, section

section("strings")

ALL = ("strings_ascii", "strings_unicode", "strings_null_heavy", "strings_pattern")
PLAIN = ("strings_ascii", "strings_unicode", "strings_pattern")
NULLS = ("strings_null_heavy",)

# ---------------------------------------------------------------------------
# Length and case
# ---------------------------------------------------------------------------

case(
    "strings/len",
    "str.len",
    frames=ALL,
    expr=lambda pd, df: df["value"].str.len(),
    note="characters and not bytes, which the unicode frame is the whole reason for",
)
for name in (
    "lower",
    "upper",
    "title",
    "capitalize",
    "swapcase",
    "casefold",
):
    case(
        f"strings/{name}",
        f"str.{name}",
        frames=ALL,
        expr=(lambda method: lambda pd, df: getattr(df["value"].str, method)())(name),
        note="case folding is locale free and it is not one to one, so a Turkish i and "
        "a German sharp s are both here in the unicode frame",
    )

# ---------------------------------------------------------------------------
# Trimming and padding
# ---------------------------------------------------------------------------

for name in ("strip", "lstrip", "rstrip"):
    case(
        f"strings/{name}",
        f"str.{name}",
        frames=ALL,
        expr=(lambda method: lambda pd, df: getattr(df["value"].str, method)())(name),
    )
    case(
        f"strings/{name}-chars",
        f"str.{name}",
        level="L3",
        covers=("to_strip",),
        frames=PLAIN,
        expr=(lambda method: lambda pd, df: getattr(df["value"].str, method)("ab"))(name),
        note="a set of characters and not a prefix, which is the thing everybody has "
        "been bitten by at least once",
    )

case(
    "strings/pad-left",
    "str.pad",
    level="L3",
    covers=("width", "side"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.pad(12, side="left"),
)
case(
    "strings/pad-both",
    "str.pad",
    level="L3",
    covers=("width", "side", "fillchar"),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.pad(12, side="both", fillchar="."),
    note="an odd amount of padding goes somewhere, and which side gets the extra "
    "character is not written down anywhere except in the implementation",
)
case(
    "strings/center",
    "str.center",
    level="L3",
    covers=("width",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.center(12),
)
case(
    "strings/ljust",
    "str.ljust",
    level="L3",
    covers=("width",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.ljust(10),
)
case(
    "strings/rjust",
    "str.rjust",
    level="L3",
    covers=("width",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.rjust(10),
)
case(
    "strings/zfill",
    "str.zfill",
    level="L3",
    covers=("width",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.zfill(10),
    note="zfill puts the zeros after a leading sign, which is a rule inherited from "
    "Python and not from anything about strings",
)
case(
    "strings/wrap",
    "str.wrap",
    level="L3",
    covers=("width",),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.wrap(5),
)
case(
    "strings/repeat",
    "str.repeat",
    level="L3",
    covers=("repeats",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.repeat(3),
)

# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

case(
    "strings/contains",
    "str.contains",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.contains("a"),
    note="a null gives a null and not a false, and the result is a nullable boolean "
    "because of it, which changes what a mask built from this does",
)
case(
    "strings/contains-regex-false",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.contains(".", regex=False),
    note="the default is a regular expression, so a literal dot needs saying so",
)
case(
    "strings/contains-case-false",
    "str.contains",
    level="L3",
    covers=("pat", "case"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.contains("A", case=False),
)
case(
    "strings/contains-na",
    "str.contains",
    level="L3",
    covers=("pat", "na"),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.contains("a", na=False),
)
case(
    "strings/startswith",
    "str.startswith",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.startswith("a"),
)
case(
    "strings/endswith",
    "str.endswith",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.endswith("z"),
)
case(
    "strings/startswith-tuple",
    "str.startswith",
    level="L3",
    covers=("pat",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.startswith(("a", "b")),
)
case(
    "strings/find",
    "str.find",
    level="L3",
    covers=("sub",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.find("a"),
    note="minus one when it is not there, which is the C answer and not the Python exception one",
)
case(
    "strings/rfind",
    "str.rfind",
    level="L3",
    covers=("sub",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.rfind("a"),
)
case(
    "strings/count",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.count("a"),
)
case(
    "strings/count-regex",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.count(r"\d"),
)
case(
    "strings/match",
    "str.match",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.match(r"[a-z]+"),
    note="match anchors at the start and not at the end, which is the difference "
    "between it and fullmatch and the reason both exist",
)
case(
    "strings/fullmatch",
    "str.fullmatch",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.fullmatch(r"[a-z]+"),
)
case(
    "strings/findall",
    "str.findall",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.findall(r"\d+"),
)
case(
    "strings/extract",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z]+)(\d+)"),
    note="two groups gives two columns named zero and one, and a row that does not "
    "match gives nulls rather than being dropped",
)
case(
    "strings/extract-named",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"(?P<letters>[a-z]+)"),
)
case(
    "strings/extract-expand-false",
    "str.extract",
    level="L3",
    covers=("pat", "expand"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z]+)", expand=False),
    note="one group and expand off gives a Series rather than a one column frame, "
    "which is a different return type from the same call",
)
case(
    "strings/extractall",
    "str.extractall",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extractall(r"(\d)"),
    note="a two level index with a match number in it, which is the only place in the "
    "string accessor that a row count changes",
)

# ---------------------------------------------------------------------------
# Slicing and splitting
# ---------------------------------------------------------------------------

case(
    "strings/slice",
    "str.slice",
    level="L3",
    covers=("start", "stop"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.slice(1, 4),
)
case(
    "strings/slice-step",
    "str.slice",
    level="L3",
    covers=("start", "stop", "step"),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.slice(None, None, 2),
)
case(
    "strings/slice-negative",
    "str.slice",
    level="L3",
    covers=("start",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.slice(-3),
    note="a slice past the end of a short string is empty and not an error",
)
case(
    "strings/slice-replace",
    "str.slice_replace",
    level="L3",
    covers=("start", "stop", "repl"),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.slice_replace(1, 3, "XX"),
)
case(
    "strings/get",
    "str.get",
    level="L3",
    covers=("i",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.get(1),
    note="past the end gives a null rather than raising, which is not what Python does",
)
case(
    "strings/split",
    "str.split",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.split("-"),
)
case(
    "strings/split-expand",
    "str.split",
    level="L3",
    covers=("pat", "expand"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.split("-", expand=True),
    note="the column count is the widest row, and every shorter row is padded with "
    "nulls, so one long row changes the shape of the whole answer",
)
case(
    "strings/split-n",
    "str.split",
    level="L3",
    covers=("pat", "n"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.split("-", n=1),
)
case(
    "strings/rsplit",
    "str.rsplit",
    level="L3",
    covers=("pat", "n"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.rsplit("-", n=1),
)
case(
    "strings/split-whitespace",
    "str.split",
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.split(),
    note="no pattern means split on any run of whitespace and drop the empties, which "
    "is a different algorithm from splitting on a single space",
)
case(
    "strings/partition",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("-"),
)
case(
    "strings/rpartition",
    "str.rpartition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.rpartition("-"),
)
case(
    "strings/join",
    "str.join",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.split("-").str.join("+"),
)

# ---------------------------------------------------------------------------
# Replacing and concatenating
# ---------------------------------------------------------------------------

case(
    "strings/replace-literal",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.replace("a", "A", regex=False),
)
case(
    "strings/replace-regex",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.replace(r"\d+", "N", regex=True),
)
case(
    "strings/replace-n",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "n", "regex"),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.replace("a", "A", n=1, regex=False),
)
case(
    "strings/replace-backreference",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.replace(r"([a-z])(\d)", r"\2\1", regex=True),
)
case(
    "strings/removeprefix",
    "str.removeprefix",
    level="L3",
    covers=("prefix",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.removeprefix("a"),
)
case(
    "strings/removesuffix",
    "str.removesuffix",
    level="L3",
    covers=("suffix",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.removesuffix("z"),
)
case(
    "strings/cat-scalar",
    "str.cat",
    level="L3",
    covers=("sep",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.cat(sep="|"),
    note="with no others this reduces the whole column to one string, and a null makes "
    "the whole thing vanish unless na_rep says otherwise",
)
case(
    "strings/cat-na-rep",
    "str.cat",
    level="L3",
    covers=("sep", "na_rep"),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.cat(sep="|", na_rep="?"),
)
case(
    "strings/cat-others",
    "str.cat",
    level="L3",
    covers=("others", "sep"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.cat(df["value"].str.upper(), sep="/"),
)

# ---------------------------------------------------------------------------
# The predicates
# ---------------------------------------------------------------------------

for name in (
    "isalpha",
    "isnumeric",
    "isalnum",
    "isdigit",
    "isdecimal",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "isascii",
):
    case(
        f"strings/{name}",
        f"str.{name}",
        frames=ALL,
        expr=(lambda method: lambda pd, df: getattr(df["value"].str, method)())(name),
        note="digit, decimal and numeric are three different questions and the unicode "
        "frame carries a character that answers them differently",
    )

case(
    "strings/normalize-nfc",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.normalize("NFC"),
    note="the unicode frame carries the same visible character in both composed and "
    "decomposed form, so this case is the one that says they are still different "
    "strings until somebody normalizes them",
)
case(
    "strings/normalize-nfd",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.normalize("NFD"),
)
case(
    "strings/normalize-len",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.normalize("NFD").str.len(),
    note="normalizing changes the length, which is the point",
)
case(
    "strings/get-dummies",
    "str.get_dummies",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.get_dummies(sep="-"),
)
case(
    "strings/translate",
    "str.translate",
    level="L3",
    covers=("table",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.translate(str.maketrans("abc", "xyz")),
)

# ---------------------------------------------------------------------------
# Strings outside the accessor
# ---------------------------------------------------------------------------

case(
    "strings/sort",
    "Series.sort_values",
    frames=ALL,
    expr=lambda pd, df: df["value"].sort_values(),
    note="code point order and not any locale's order, and the unicode frame is what "
    "makes those two different",
)
case(
    "strings/compare",
    "Series.lt",
    frames=ALL,
    expr=lambda pd, df: df["value"] < "m",
)
case(
    "strings/max",
    "Series.max",
    frames=ALL,
    expr=lambda pd, df: df["value"].max(),
)
case(
    "strings/value-counts",
    "Series.value_counts",
    frames=("strings_pattern", "strings_null_heavy"),
    expr=lambda pd, df: df["value"].value_counts().sort_index(),
)
