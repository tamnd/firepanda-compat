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
