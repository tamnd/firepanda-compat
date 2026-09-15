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
FOLDING = ("strings_folding",)
CASED = ("strings_ascii", "strings_pattern", "strings_null_heavy", "strings_folding")

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
    frames=CASED,
    expr=lambda pd, df: df["value"].str.contains("A", case=False),
    note="an ASCII pattern, which every plausible fold agrees about, so this one is "
    "the baseline and the three below it are the ones that can tell the folds apart",
)
case(
    "strings/contains-case-false-longer",
    "str.contains",
    level="L3",
    covers=("pat", "case"),
    frames=FOLDING,
    expr=lambda pd, df: df["value"].str.contains("ss", case=False),
    note="the pattern that separates a search from casefold, since casefold turns a "
    "sharp s into two letters and would find this in five rows a search does not",
)
case(
    "strings/contains-case-false-long-s",
    "str.contains",
    level="L3",
    covers=("pat", "case"),
    frames=FOLDING,
    expr=lambda pd, df: df["value"].str.contains("ſ", case=False),
    note="long s, which is the pattern that separates a search from both of the other "
    "two rules at once, since casefold gets two rows wrong here and the lower case "
    "gets seven wrong, and a search folds it onto a plain s",
)
case(
    "strings/contains-case-false-micro",
    "str.contains",
    level="L3",
    covers=("pat", "case"),
    frames=FOLDING,
    expr=lambda pd, df: df["value"].str.contains("μm", case=False),
    note="the micro sign against Greek mu, which is the same separation as the sigma "
    "and is here because it is the lowest code point in the whole fold table",
)
case(
    "strings/match-case-false",
    "str.match",
    level="L3",
    covers=("pat", "case"),
    frames=CASED,
    expr=lambda pd, df: df["value"].str.match("straße", case=False),
)
case(
    "strings/fullmatch-case-false",
    "str.fullmatch",
    level="L3",
    covers=("pat", "case"),
    frames=CASED,
    expr=lambda pd, df: df["value"].str.fullmatch("straße", case=False),
    note="a folded match can cover a different number of bytes than the pattern, since "
    "long s is two bytes and is compared as the one byte s, so this cannot be decided "
    "by comparing lengths the way the case sensitive one can",
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
    "strings/count-anchor",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.count(r"^[a-z]"),
    note="the rest of the row becomes the text after every match, so the start of "
    "the text moves with the scan and a row of twenty letters holds twenty matches "
    "of a pattern anchored at the start. Python's re module answers one for every "
    "row here, so this is the case that says which of the two readings pandas has, "
    "and the ascii frame is the one whose rows are the same letters at every length",
)
case(
    "strings/count-boundary",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.count(r"\b"),
    note="a boundary has no width and is found ahead of the scan, which counts it "
    "where it was found and again from there, so a scan that stepped past an empty "
    "match undercounts and a scan that stepped one byte after every empty match "
    "overcounts. The row of five hundred letters answers five hundred, which is the "
    "one row where getting the rule wrong is off by hundreds rather than by one",
)
case(
    "strings/count-empty-match",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.count("[q]*"),
    note="the scan steps a byte at a time rather than a character at a time, so a "
    "pattern that can match nothing counts the bytes of a row and one more. This is "
    "the other half of strings/count-empty, which asks the same question of the "
    "literal path, and the two are separate cases because a pattern with a "
    "metacharacter in it goes to the regular expression engine and takes different "
    "code to the same answer",
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
    "strings/contains-regex",
    "str.contains",
    level="L3",
    covers=("pat", "regex"),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.contains(r"[0-9]+"),
    note="the corpus had a literal contains and a regex match and fullmatch and no "
    "regex contains at all, which left the one method of the three that needs no "
    "rewrite as the one with nothing regular expression shaped pointed at it",
)
case(
    "strings/match-alternation",
    "str.match",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.match("a|b"),
    note="match anchors the whole pattern and not its first arm, because pandas "
    "answers it by wrapping the pattern in a group before it puts the anchor on, so "
    "a row holding a b later and starting with neither letter is False here and "
    "would be True under the anchor without the group",
)
case(
    "strings/fullmatch-flag-group",
    "str.fullmatch",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.fullmatch("(?m)[a-z]+"),
    note="the anchors fullmatch adds are the ends of the row and not the ends of a "
    "line even though the pattern asked for m, because pandas adds them outside the "
    "group the flag sits in, and the row holding a newline is the only row in the "
    "corpus that can tell those two readings apart",
)
case(
    "strings/match-literal",
    "str.match",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.match("a"),
    note="the same pattern the contains and fullmatch cases use, because the three "
    "questions differ only in where the pattern is allowed to sit and a case that "
    "uses a different pattern for each of them cannot show that",
)
case(
    "strings/fullmatch-literal",
    "str.fullmatch",
    level="L3",
    covers=("pat",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.fullmatch("a"),
    note="the ascii frame holds a row that is exactly this pattern, which is the "
    "only row in the corpus where fullmatch and match disagree",
)
case(
    "strings/count-empty",
    "str.count",
    level="L3",
    covers=("pat",),
    frames=("strings_unicode", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.count(""),
    note="counted in bytes rather than in characters, because pandas answers this "
    "out of Arrow and Arrow counts a match at every byte offset and once past the "
    "end, so a row holding one accented letter answers one more than Python's re "
    "module does for the same row",
)
case(
    "strings/findall",
    "str.findall",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.findall(r"\d+"),
)
EXTRACT_LABELS = (
    "and the whole frame is compared here, so this case carries "
    "`engine/integer-column-labels`: pandas labels an unnamed group with its own "
    "position as an integer and firepanda labels it with the text of that integer. "
    "The two cases under this one read a column out by position instead, which is "
    "where the values get scored"
)

case(
    "strings/extract",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z]+)(\d+)"),
    note="two groups gives two columns named zero and one, and a row that does not "
    "match gives nulls rather than being dropped, " + EXTRACT_LABELS,
)
case(
    "strings/extract-first",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z]+)(\d+)").iloc[:, 0].rename("value"),
    note="the same two groups as the case above read a column at a time, so the values "
    "are compared without the labels. `abc123` matches and `barfoo` does not, and a row "
    "that does not match is null in this column and in the one beside it, which is what "
    "the pair of these two says that neither says alone",
)
case(
    "strings/extract-second",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z]+)(\d+)").iloc[:, 1].rename("value"),
)
case(
    "strings/extract-optional-group",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"([a-z])(\d)?").iloc[:, 1].rename("value"),
    note="the one case where the two columns of a row disagree about whether there was "
    "a match. `abc123` matches with the optional group left out, because the letter and "
    "the digit are not next to each other, so the first column holds a letter and this "
    "one is null on the same row. An implementation that wrote the row's match state "
    "across the width passes every other extract case on the board and fails this one",
)
case(
    "strings/extract-search-not-anchored",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.extract(r"(\d+)").iloc[:, 0].rename("value"),
    note="pandas runs search here and match for `str.match`, so a pattern with no "
    "anchor finds its match in the middle of a row. `abc123` answers `123` and the "
    "three mask methods on this accessor answer False for the same pattern on the same "
    "row, which is the pair of behaviours this case pins",
)
case(
    "strings/extract-unicode-word",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.extract(r"(\w)").iloc[:, 0].rename("value"),
    note="this is one of the three names on the accessor that never reach Arrow, so "
    "`\\w` is read as Python reads it and is 138558 code points rather than the 63 of "
    "ASCII that `str.count` gets for the same letter on the same column. The Arabic and "
    "the Chinese rows are where the two readings part, and the row holding a space "
    "matches under neither",
)
case(
    "strings/extract-unicode-digit",
    "str.extract",
    level="L3",
    covers=("pat",),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.extract(r"(\d)").iloc[:, 0].rename("value"),
    note="Python reads `\\d` as exactly what `str.isdecimal` accepts, which is not what "
    "`str.isdigit` accepts and is 128 code points narrower. The mathematical double "
    "struck digits are decimal and match, and the half sign and the Roman eight are not "
    "and do not, which is three rows no ASCII frame can tell apart",
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
CUT_BY_POSITION = (
    "The column is read by position and then renamed, because the name pandas gives it "
    "is the integer label and that is `engine/integer-column-labels`, which the two "
    "frame cases above carry. Renaming here is what keeps this case about the values"
)

CUT_LABELS = (
    "and the whole frame is compared here, so this case carries "
    "`engine/integer-column-labels`: pandas labels the three columns with the integers "
    "0, 1 and 2 and firepanda labels them with the text. The five cases under this one "
    "read a column out by position instead, which is where the values get scored"
)

case(
    "strings/partition",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("-"),
    note="no row of this frame holds a hyphen, which is on purpose rather than by "
    "accident, because a row the separator is not in is where the two names differ in "
    "the way nobody guesses: the row survives whole and goes into the first column for "
    "this name and into the third for the other one, " + CUT_LABELS,
)
case(
    "strings/rpartition",
    "str.rpartition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.rpartition("-"),
    note="the same eighteen rows and the same absent separator, and the pair is what "
    "says the two names put an uncut row at opposite ends, " + CUT_LABELS,
)
case(
    "strings/partition-head",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("o").iloc[:, 0].rename("value"),
    note="a separator that is really in three of the rows, read a column at a time so "
    "the values are compared without the labels. `foofoobar` holds the separator three "
    "times, which is what makes this case and its `r` twin disagree with each other and "
    "is the only way to tell that a search really ran. " + CUT_BY_POSITION,
)
case(
    "strings/partition-sep",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("o").iloc[:, 1].rename("value"),
    note="the middle column, which is the separator where it was found and the empty "
    "string where it was not, and is the one of the three that cannot tell the two "
    "names apart since they find the same separator wherever they find it. " + CUT_BY_POSITION,
)
case(
    "strings/partition-tail",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.partition("o").iloc[:, 2].rename("value"),
    note="the part after the cut, which for this name is everything left over and for "
    "a row with no separator in it is empty. " + CUT_BY_POSITION,
)
case(
    "strings/rpartition-head",
    "str.rpartition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.rpartition("o").iloc[:, 0].rename("value"),
    note="the same column of the other name, and the answer is different on every row "
    "that holds the separator more than once, which is what says the search started "
    "from the right end. " + CUT_BY_POSITION,
)
case(
    "strings/rpartition-tail",
    "str.rpartition",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.rpartition("o").iloc[:, 2].rename("value"),
    note="the column an uncut row lands in for this name, so the empty strings here "
    "and the empty strings in `strings/partition-tail` are in different rows. " + CUT_BY_POSITION,
)
case(
    "strings/partition-null",
    "str.partition",
    level="L3",
    covers=("sep",),
    frames=("strings_null_heavy",),
    expr=lambda pd, df: df["value"].str.partition("v").iloc[:, 2].rename("value"),
    note="two thirds of this frame is missing and the rest is either the empty string "
    "or a row the separator starts, so it asks three questions at once: a missing row "
    "is missing in all three columns rather than empty in any of them, an empty row is "
    "not a missing one, and a separator at the very start leaves the first column empty "
    "rather than dropping the row. " + CUT_BY_POSITION,
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
    "strings/replace-anchor",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_ascii",),
    expr=lambda pd, df: df["value"].str.replace(r"^[a-z]", "#", regex=True),
    note="the same pattern as strings/count-anchor on the same frame, and the two "
    "of them are on the board to say that Arrow answers it twice differently. The "
    "count cuts the row after every match so the start of the text moves with the "
    "scan and a row of twenty letters holds twenty matches. The replace does not "
    "cut it, so the start of the text stays where it was and only the first letter "
    "of a row is swapped. Nothing anywhere says the two loops disagree and both "
    "readings look right on their own, so this case is the measurement",
)
case(
    "strings/replace-empty-match",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_unicode",),
    expr=lambda pd, df: df["value"].str.replace("[q]*", "-", regex=True),
    note="the other half of strings/count-empty-match, and the second place the "
    "two loops disagree. The count steps a byte at a time and answers the bytes of "
    "a row and one more. The replace steps a character at a time and writes a "
    "marker before every character and one after the last, so the unicode frame "
    "answers a different number of markers from the number the count answers on "
    "the very same row",
)
case(
    "strings/replace-boundary",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.replace(r"\b", "#", regex=True),
    note="the third place the two loops disagree, and the one that reads as a bug "
    "until it is measured. A match of no width landing exactly where the last match "
    "ended is thrown away, and one character is copied across rather than the scan "
    "simply stepping, so replacing a boundary in a word marks the two ends of the "
    "word and nothing in the middle. The count of the same pattern on the same "
    "frame answers a number per row instead. The row of five hundred letters is the "
    "one where a scan that dropped the rule writes a row hundreds of characters "
    "longer than the right one",
)
case(
    "strings/replace-whole-match",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "regex"),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.replace(r"\d+", r"[\0]", regex=True),
    note="the replacement has a grammar and it is RE2's rather than Python's, "
    "where a backslash and a zero is the whole match and Python has no such thing "
    "at all. strings/replace-backreference beside this one is the numbered groups "
    "and this is the reference that needs no group in the pattern, so an "
    "implementation that only kept slots when a group was written fails here and "
    "passes there",
)
case(
    "strings/replace-n",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "n", "regex"),
    frames=("strings_pattern", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.replace("a", "A", n=1, regex=False),
    note="the pattern frame is here because the ascii frame cannot tell this case "
    "from strings/replace-literal. Its rows are the alphabet cut at every length, "
    "so no row of it holds the letter a twice and an implementation that ignored n "
    "entirely would pass on it",
)
case(
    "strings/replace-empty",
    "str.replace",
    level="L3",
    covers=("pat", "repl"),
    frames=("strings_unicode", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.replace("", "-"),
    note="the other half of strings/count-empty, and the two of them together are "
    "the reason both are on the board. An empty pattern is counted in characters "
    "here and in bytes there, in the same accessor on the same row, because "
    "pyarrow's replace_substring does not terminate on an empty pattern and pandas "
    "has a guard that hands that one case to Python while count has no guard and "
    "stays in Arrow",
)
case(
    "strings/replace-case-false",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "case"),
    frames=CASED,
    expr=lambda pd, df: df["value"].str.replace("straße", "#", case=False),
    note="the one of the four case insensitive names pandas does not answer out of "
    "Arrow, since it refuses case=False there and falls back to a Python path that "
    "escapes the pattern and runs it with re.IGNORECASE. The two rules agree on every "
    "pair, which did not have to be true, and this is the case that says so",
)
case(
    "strings/replace-case-false-n-zero",
    "str.replace",
    level="L3",
    covers=("pat", "repl", "n", "case"),
    frames=("strings_folding", "strings_ascii"),
    expr=lambda pd, df: df["value"].str.replace("A", "#", n=0, case=False),
    note="n=0 means no replacements to this method and every replacement to this "
    "method with case=False, because the Arrow path takes the number at its word and "
    "the fallback hands it to re.sub where a count of zero has meant unlimited for "
    "thirty years. Same method, same column, two answers, and strings/replace-n "
    "beside it is the half that reads the zero the other way",
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
CAT_FOLDS = (
    "with no others this is not a transformation at all, it folds the whole column "
    "into one string, which is the only answer on this accessor narrower than a column"
)
CAT_DROPS = (
    "a missing row is dropped rather than blanked and dropped takes its separator with "
    "it, so the separator count goes down with the row, which is the one thing a join "
    "written the obvious way gets wrong"
)

case(
    "strings/cat-scalar",
    "str.cat",
    level="L3",
    covers=("sep",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.cat(sep="|"),
    note="these frames hold no nulls, so this one scores the separator and the order "
    "and nothing about a missing row, which is what cat-drops-null is for. " + CAT_FOLDS,
)
case(
    "strings/cat-default",
    "str.cat",
    level="L2",
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.cat(),
    note="sep=None is the empty string rather than a missing argument, so a bare call "
    "runs the rows together with nothing between them",
)
case(
    "strings/cat-drops-null",
    "str.cat",
    level="L3",
    covers=("sep",),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.cat(sep="|"),
    note=CAT_DROPS + ". This frame is two thirds null with an empty string beside the "
    "nulls, so it also separates the row that is dropped from the row that is empty and "
    "kept, which come out looking alike and are reached by opposite rules",
)
case(
    "strings/cat-na-rep",
    "str.cat",
    level="L3",
    covers=("sep", "na_rep"),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.cat(sep="|", na_rep="?"),
    note="given a stand in the missing row is a row again and its separator comes back, "
    "so this answer is longer than cat-drops-null by more than the text",
)
case(
    "strings/cat-na-rep-empty",
    "str.cat",
    level="L3",
    covers=("sep", "na_rep"),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.cat(sep="|", na_rep=""),
    note="pandas reads whether to drop the row off whether na_rep was given and not off "
    "what it holds, so an empty stand in keeps the row and its separator and drops only "
    "the text, which is a different answer from leaving the argument out",
)
case(
    "strings/cat-others",
    "str.cat",
    level="L3",
    covers=("others", "sep"),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.cat(df["value"].str.upper(), sep="/"),
    note="the other half of this name, which aligns the two columns on their labels "
    "before concatenating and so is a different operation from the fold above rather "
    "than the same one with an argument",
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
        "frame carries a row for each of the three answers they can give, which it did "
        "not until these arms were written and the corpus was checked rather than trusted",
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
    "strings/normalize-nfkc",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.normalize("NFKC"),
    note="the wider relation, which the unicode frame is full of on purpose: the fi "
    "ligature becomes two letters, the half sign becomes three characters, the "
    "roman eight becomes four ordinary letters and the titlecase digraph becomes "
    "two, so a library answering the canonical form here is wrong on four rows, and "
    "the other three frames are here because they are the identity under all four "
    "forms and a library that normalized an ascii row or a missing one would be "
    "wrong on them without any character in the frame being unusual",
)
case(
    "strings/normalize-nfkd",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=ALL,
    expr=lambda pd, df: df["value"].str.normalize("NFKD"),
    note="the fourth form, and it is not the third with a different last step on "
    "every row, since the titlecase digraph gives a caron of its own here and a z "
    "with a caron under the form above, which is the difference that catches an "
    "implementation treating the K forms as a decomposition difference and nothing else",
)
case(
    "strings/normalize-folding",
    "str.normalize",
    level="L3",
    covers=("form",),
    frames=FOLDING,
    expr=lambda pd, df: df["value"].str.normalize("NFKC"),
    note="the folding frame separates the four forms as well as the unicode one does "
    "and separates them on different rows, since its long s becomes an ordinary s, "
    "its micro sign becomes a greek mu and all three of its digraphs come apart, and "
    "none of those is a difference the unicode frame's accented rows can score",
)
case(
    "strings/get-dummies",
    "str.get_dummies",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.get_dummies(sep="-"),
    note="no row in this frame holds a hyphen, so nothing here splits and every "
    "row is one whole token, which is the path where the answer is one column "
    "per distinct value and is worth scoring on its own because it is the one "
    "an implementation that dropped rows without the separator would get wrong, "
    "and it still scores the byte order of the labels and the empty row's empty "
    "label, but the splitting is scored by the case below rather than by this one",
)
case(
    "strings/get-dummies-split",
    "str.get_dummies",
    level="L3",
    covers=("sep",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.get_dummies(sep="a"),
    note="a separator many rows do hold, and hold at the start and at the end and "
    "twice over, so this is where the empty token gets in by the three routes that "
    "are not an empty row, and where a row cut into three pieces has to contribute "
    "all three rather than the first and the last",
)
case(
    "strings/get-dummies-null",
    "str.get_dummies",
    level="L3",
    covers=("sep",),
    frames=NULLS,
    expr=lambda pd, df: df["value"].str.get_dummies(sep="v"),
    note="two thirds of this frame is missing and the rest is an empty string or "
    "a value starting with the separator, so a missing row has to contribute no "
    "label and come back as zeros where an empty row has to contribute the empty "
    "label and come back as a one under it, which is the single place on this "
    "accessor where the two do not behave alike",
)
case(
    "strings/get-dummies-bool",
    "str.get_dummies",
    level="L3",
    covers=("dtype",),
    frames=("strings_pattern",),
    expr=lambda pd, df: df["value"].str.get_dummies(sep="a", dtype=bool),
    note="the same answer read as flags rather than as the int64 pandas gives by "
    "default, which is the shape the data really has since the only two values "
    "this method can produce are a one and a zero",
)
case(
    "strings/translate",
    "str.translate",
    level="L3",
    covers=("table",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.translate(str.maketrans("abc", "xyz")),
    note="the plain form, whose table maps three letters onto three that are not "
    "themselves keys, so an implementation running the table entry by entry would "
    "give the same answer and this case could not tell, which is what the swap is for",
)
case(
    "strings/translate-swap",
    "str.translate",
    level="L3",
    covers=("table",),
    frames=PLAIN,
    expr=lambda pd, df: df["value"].str.translate(str.maketrans("ab", "ba")),
    note="a table whose values are also its keys, which is the difference between "
    "this name and replace with a mapping, since every key is applied in the same "
    "pass and so this really swaps the two letters where replace would collapse them",
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
