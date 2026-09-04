"""The cost matrix: what a correct answer costs, per operation.

Implements `docs/specs/09-resources.md`.

The stated goal for firepanda is ten times the performance on a tenth of the
resources. That is a claim about every operation, and firepanda-bench checks it on
thirty seven queries, which means it is not checked on `str.extract`, on `reindex`,
or on any of the other thousand callables a real program calls. A user whose program
is sixty percent string work finds that out on their own.

The conformance suite already calls every operation on known inputs, with the answers
already verified. This module puts a timer and a memory high water mark on the same
operations and produces a row per operation instead of a row per query. It is the
bridge between the two repositories: firepanda-bench answers "is this query fast" and
this answers "which operation is slow", which is the question you need answered before
you can fix anything.

Four rules it is built around.

**One process per engine per operation.** Peak resident set is a high water mark on a
process. Two engines in one interpreter cannot both be measured, and neither can two
operations, because the second one inherits the first one's high water mark and reads
as expensive for something another operation did. This is the same rule
firepanda-bench learned and it is not negotiable.

**The answer is consumed before the timer stops.** Anything else measures a lazy
engine's ability to defer work rather than its ability to do it. firepanda is lazy
underneath after M4, so this is not a theoretical concern here.

**Seven repeats, the median published and the interquartile range published beside
it.** A single number with no spread is not a measurement, and a two times speedup
that is inside the noise should look like one.

**Publish the rows we lose.** A cost matrix where firepanda wins every row has either
been curated or is measuring the wrong thing, and the first person to notice will be
somebody deciding whether to trust the project. A row below one is a performance bug
with a name and an input size, and this table is where it gets its name.

The budget corpus is not the correctness corpus. That one is small and mean by design,
and timing a call on 64 rows measures interpreter overhead. This one is the same
generator, the same constants and the same seed at a size where the work is the work:
one million rows by default and ten million under `--large`, with no edge value
placement, because a denormal does not change a runtime and the corpus should stay
comparable between runs. It goes to Arrow IPC under `corpus/budget/<rows>/` and is
loaded through the same engine method the correctness corpus uses, so a Mojo driver
needs nothing new to run it.

Usage:
    python -m fpcompat.budget --corpus                 # build the budget corpus
    python -m fpcompat.budget                          # measure pandas
    python -m fpcompat.budget --engine firepanda       # measure the subject
    python -m fpcompat.budget --matrix                 # the table, from the results
    python -m fpcompat.budget --check --rows 10000     # the corpus digests, in CI
    python -m fpcompat.budget --operations             # the table other repos read
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from fpcompat import corpus

ROOT = corpus.ROOT
RESULTS = ROOT / "results"
BUDGET = corpus.CORPUS / "budget"
MANIFEST = ROOT / "corpus" / "budget-manifest.json"
TABLE = ROOT / "operations.json"

# `ru_maxrss` is kilobytes on Linux and bytes on macOS. Getting this wrong makes a
# memory column wrong by a factor of 1024 on one platform, which is large enough that
# somebody would catch it and small enough to survive review.
RSS_UNIT = 1 if sys.platform == "darwin" else 1024

DEFAULT_ROWS = 1_000_000
LARGE_ROWS = 10_000_000
REPEATS = 7

# The group cardinality of the keyed frames. A thousand groups is the shape where a
# hash aggregation is doing real work and the table of groups still fits in cache,
# which is the interesting middle. The ten group and the unique cases are separate
# operations rather than a separate corpus.
GROUPS = 1_000

# The frames the budget corpus has. Written down rather than derived, so that asking
# whether the corpus is on disk does not mean building it first.
FRAMES = ("keyed", "numeric", "strings", "temporal", "lookup")

# How many rows from each end the manifest digest covers. Hashing every value the way
# the correctness manifest does costs a `to_pylist` of ten million values per column,
# which is a minute of CI time to catch a change that shows up in the first row. Both
# ends, because a generator change that only moves the tail is still a real change and
# reading only the head would miss it.
SAMPLE = 500


# ---------------------------------------------------------------------------
# The budget corpus
# ---------------------------------------------------------------------------


def signed(words: np.ndarray) -> np.ndarray:
    """Turns a splitmix64 stream into int64 values that are not astronomically large.

    Shifting away the top 24 bits keeps every value inside forty bits, which is small
    enough that a sum over ten million rows does not overflow and large enough that a
    sort has to look at more than one byte.

    Args:
        words: The unsigned stream.

    Returns:
        int64 values.
    """
    return (words >> np.uint64(24)).astype(np.int64)


def unit(words: np.ndarray) -> np.ndarray:
    """Turns a splitmix64 stream into float64 values in [0, 1).

    Fifty three bits, which is the mantissa, so every value is exactly representable
    and no engine gets to be faster by rounding differently.

    Args:
        words: The unsigned stream.

    Returns:
        float64 values.
    """
    return (words >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def text(draw: Callable[..., np.ndarray], rows: int) -> list[str]:
    """Builds the string column.

    Lengths between four and twenty seven bytes over an alphabet with a space in it,
    so `str.split` has something to split and `str.contains` cannot be answered by a
    length check. The characters come from one long stream that the rows read
    overlapping windows of, which builds a hundred times faster than one draw per
    character and gives the same length distribution and the same byte frequency.

    Args:
        draw: The stream builder.
        rows: How many strings.

    Returns:
        The strings.
    """
    alphabet = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz0123456789 ", dtype=np.uint8)
    lengths = corpus.below(draw("budget-len"), 24) + 4
    pool = alphabet[corpus.below(draw("budget-text", rows + 32), len(alphabet))]
    letters = pool.tobytes().decode()
    return [letters[index : index + int(length)] for index, length in enumerate(lengths)]


def frames(rows: int) -> dict[str, pa.Table]:
    """Builds the budget corpus at a given size.

    The same splitmix64 counter stream, the same seed and the same per name salt as
    the correctness corpus, so a Mojo driver can produce identical columns without
    reading a file and a person reading both does not have to work out whether a
    difference between them means something. What is different is the size and the
    absence of edge values.

    Args:
        rows: How many rows each frame gets. The lookup frame is `GROUPS` rows
            whatever this is, because it is the one row per group side of a join.

    Returns:
        Frame name to Arrow table.
    """

    def draw(name: str, count: int | None = None) -> np.ndarray:
        return corpus.splitmix64(corpus.SEED ^ corpus.salt(name), count or rows)

    built: dict[str, pa.Table] = {}

    # The db-benchmark shape, which is what most grouped and joined work looks like.
    built["keyed"] = pa.table(
        {
            "id1": pa.array(
                [f"id{value:03d}" for value in corpus.below(draw("budget-id1"), 100)],
                type=pa.large_string(),
            ),
            "id2": pa.array(corpus.below(draw("budget-id2"), GROUPS), type=pa.int64()),
            "id3": pa.array(corpus.below(draw("budget-id3"), rows), type=pa.int64()),
            "v1": pa.array(corpus.below(draw("budget-v1"), 100), type=pa.int64()),
            "v2": pa.array(unit(draw("budget-v2")), type=pa.float64()),
        }
    )

    # Fixed width numeric work: casts, comparisons, reductions and sorts, which is
    # where the ceiling is the memory controller rather than the language.
    built["numeric"] = pa.table(
        {
            "a": pa.array(signed(draw("budget-a")), type=pa.int64()),
            "b": pa.array(signed(draw("budget-b")), type=pa.int64()),
            "c": pa.array(unit(draw("budget-c")), type=pa.float64()),
        }
    )

    # Strings that look like strings. Fixed width filler would make every string
    # operation measure the same memcpy, and the length distribution is most of what a
    # string kernel's cost is about.
    built["strings"] = pa.table(
        {
            "s": pa.array(text(draw, rows), type=pa.large_string()),
            "key": pa.array(corpus.below(draw("budget-skey"), GROUPS), type=pa.int64()),
        }
    )

    # One second apart, so a floor and a resample have something to do.
    start = np.int64(1_600_000_000) * np.int64(1_000_000_000)
    built["temporal"] = pa.table(
        {
            "t": pa.array(
                start + np.arange(rows, dtype=np.int64) * np.int64(1_000_000_000),
                type=pa.timestamp("ns"),
            ),
            "v": pa.array(signed(draw("budget-tv")), type=pa.int64()),
        }
    )

    # The right hand side of a join, one row per group.
    built["lookup"] = pa.table(
        {
            "id2": pa.array(np.arange(GROUPS, dtype=np.int64), type=pa.int64()),
            "w": pa.array(signed(draw("budget-w", GROUPS)), type=pa.int64()),
        }
    )
    return built


def path_of(rows: int, name: str) -> Path:
    """Where one budget frame lives.

    Args:
        rows: The corpus size.
        name: The frame name.

    Returns:
        The path.
    """
    return BUDGET / str(rows) / f"{name}.arrow"


def frame_name(rows: int, name: str) -> str:
    """The name an engine loads a budget frame by.

    Both engines resolve a frame name against the corpus directory, so a name with the
    size in it addresses the budget corpus through the method the correctness corpus
    already uses. That is on purpose. Two corpora and one loader means the Mojo driver
    needs nothing new, and a second loader would be a second place for the two engines
    to disagree about what they were handed.

    Args:
        rows: The corpus size.
        name: The frame name.

    Returns:
        The name to pass to `Engine.frame`.
    """
    return f"budget/{rows}/{name}"


def write(rows: int) -> dict[str, pa.Table]:
    """Builds the budget corpus and writes it to Arrow IPC.

    Args:
        rows: The corpus size.

    Returns:
        The tables, so a caller that wants to digest them does not build them twice.
    """
    built = frames(rows)
    (BUDGET / str(rows)).mkdir(parents=True, exist_ok=True)
    for name, table in built.items():
        with (
            pa.OSFile(str(path_of(rows, name)), "wb") as sink,
            pa.ipc.new_file(sink, table.schema) as writer,
        ):
            writer.write_table(table)
    return built


def ready(rows: int) -> bool:
    """Whether the budget corpus at this size is already on disk.

    Args:
        rows: The corpus size.

    Returns:
        True when every frame is there.
    """
    return all(path_of(rows, name).exists() for name in FRAMES)


def sample_digest(table: pa.Table) -> str:
    """Digests a frame's schema, its row count and both of its ends.

    Args:
        table: The table.

    Returns:
        A hexadecimal SHA-256.
    """
    sha = hashlib.sha256()
    for field in table.schema:
        sha.update(f"{field.name}:{field.type}\n".encode())
    sha.update(f"rows:{table.num_rows}\n".encode())
    for column in table.columns:
        head = column.slice(0, min(SAMPLE, table.num_rows)).to_pylist()
        tail = column.slice(max(0, table.num_rows - SAMPLE)).to_pylist()
        for value in head + tail:
            sha.update(f"{value!r}\n".encode())
    return sha.hexdigest()


def describe(rows: int, built: dict[str, pa.Table] | None = None) -> dict[str, Any]:
    """Describes the budget corpus at one size, for the manifest.

    Args:
        rows: The corpus size.
        built: The tables, when the caller already has them.

    Returns:
        The manifest entry.
    """
    tables = built if built is not None else frames(rows)
    return {
        "rows": rows,
        "sample": SAMPLE,
        "frames": {
            name: {
                "rows": table.num_rows,
                "columns": [f"{field.name}:{field.type}" for field in table.schema],
                "digest": sample_digest(table),
            }
            for name, table in sorted(tables.items())
        },
    }


def read_manifest() -> dict[str, Any]:
    """Reads the committed budget manifest, or builds an empty one.

    Returns:
        The document.
    """
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {
        "seed": f"0x{corpus.SEED:016X}",
        "generator": "splitmix64 counter stream, the same one fpcompat/corpus.py uses",
        "sizes": {},
    }


# ---------------------------------------------------------------------------
# The operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """One row of the matrix.

    Attributes:
        id: The row name, which is the pandas spelling of what it does.
        section: The parity section, so a row can be read next to the conformance
            score for the same section.
        needs: Which budget frames it loads.
        covers: The pandas names it exercises, so a slow row can be traced to the
            callables it is evidence about.
        call: Takes the engine module and the loaded frames and returns the answer.
            The harness consumes the answer, not this function.
        chained: Whether this is a multi step case. Those are where a tenth of the
            memory is actually available, because peak memory in pandas is dominated
            by intermediates, and they are marked so the summary can report them apart.
    """

    id: str
    section: str
    needs: tuple[str, ...]
    covers: tuple[str, ...]
    call: Callable[[Any, dict[str, Any]], Any]
    chained: bool = False


OPERATIONS: list[Operation] = []

Frames = dict[str, Any]


def operation(
    name: str, section: str, needs: str, covers: str, chained: bool = False
) -> Callable[[Callable[[Any, Frames], Any]], Callable[[Any, Frames], Any]]:
    """Registers one operation.

    Args:
        name: The row name.
        section: The parity section.
        needs: Whitespace separated budget frame names.
        covers: Whitespace separated pandas names it exercises.
        chained: Whether it is a multi step case.

    Returns:
        The decorator.
    """

    def register(function: Callable[[Any, Frames], Any]) -> Callable[[Any, Frames], Any]:
        OPERATIONS.append(
            Operation(name, section, tuple(needs.split()), tuple(covers.split()), function, chained)
        )
        return function

    return register


# The single operations. The point of the table is that these are the calls a user
# writes without thinking about them, so they are written here the same way.


@operation("groupby.sum 1e3 groups", "groupby", "keyed", "DataFrame.groupby GroupBy.sum")
def _groupby_sum(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id2")["v1"].sum()


@operation("groupby.mean string key", "groupby", "keyed", "DataFrame.groupby GroupBy.mean")
def _groupby_mean_string(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id1")["v2"].mean()


@operation("groupby.agg two functions", "groupby", "keyed", "GroupBy.agg")
def _groupby_agg(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id2").agg({"v1": "sum", "v2": "mean"})


@operation("groupby.nunique high cardinality", "groupby", "keyed", "GroupBy.nunique")
def _groupby_nunique(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id2")["id3"].nunique()


@operation("groupby.apply lambda", "groupby", "keyed", "GroupBy.apply")
def _groupby_apply(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id1")["v1"].apply(lambda group: group.max() - group.min())


@operation("value_counts", "stats", "keyed", "Series.value_counts")
def _value_counts(pd: Any, f: Frames) -> Any:
    return f["keyed"]["id1"].value_counts()


@operation("sort_values int64", "indexing", "numeric", "DataFrame.sort_values")
def _sort_int(pd: Any, f: Frames) -> Any:
    return f["numeric"].sort_values("a")


@operation("sort_values two columns", "indexing", "keyed", "DataFrame.sort_values")
def _sort_two(pd: Any, f: Frames) -> Any:
    return f["keyed"].sort_values(["id1", "id2"])


@operation("astype int64 to int32", "basics", "numeric", "DataFrame.astype")
def _astype(pd: Any, f: Frames) -> Any:
    return f["numeric"]["a"].astype("int32")


@operation("boolean mask", "indexing", "numeric", "DataFrame.loc")
def _mask(pd: Any, f: Frames) -> Any:
    frame = f["numeric"]
    return frame[frame["a"] > 0]


@operation("isin against 50 values", "basics", "keyed", "Series.isin")
def _isin(pd: Any, f: Frames) -> Any:
    return f["keyed"]["v1"].isin(list(range(50)))


@operation("where", "basics", "numeric", "DataFrame.where")
def _where(pd: Any, f: Frames) -> Any:
    column = f["numeric"]["c"]
    return column.where(column > 0.5, 0.0)


@operation("sum", "stats", "numeric", "Series.sum")
def _sum(pd: Any, f: Frames) -> Any:
    return f["numeric"]["a"].sum()


@operation("cumsum", "stats", "numeric", "Series.cumsum")
def _cumsum(pd: Any, f: Frames) -> Any:
    return f["numeric"]["a"].cumsum()


@operation("rank", "stats", "numeric", "Series.rank")
def _rank(pd: Any, f: Frames) -> Any:
    return f["numeric"]["c"].rank()


@operation("quantile", "stats", "numeric", "Series.quantile")
def _quantile(pd: Any, f: Frames) -> Any:
    return f["numeric"]["c"].quantile([0.1, 0.5, 0.9])


@operation("describe", "stats", "numeric", "DataFrame.describe")
def _describe(pd: Any, f: Frames) -> Any:
    return f["numeric"].describe()


@operation("rolling.mean window 100", "windows", "numeric", "DataFrame.rolling Rolling.mean")
def _rolling(pd: Any, f: Frames) -> Any:
    return f["numeric"]["c"].rolling(100).mean()


@operation("str.contains literal", "strings", "strings", "str.contains")
def _contains(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.contains("ab", regex=False)


@operation("str.contains regex", "strings", "strings", "str.contains")
def _contains_regex(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.contains("a.b", regex=True)


@operation("str.upper", "strings", "strings", "str.upper")
def _upper(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.upper()


@operation("str.len", "strings", "strings", "str.len")
def _strlen(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.len()


@operation("str.split then get", "strings", "strings", "str.split str.get")
def _split(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.split(" ").str.get(0)


@operation("str.replace literal", "strings", "strings", "str.replace")
def _replace(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.replace("a", "z", regex=False)


@operation("merge on int64 key", "reshape", "keyed lookup", "DataFrame.merge")
def _merge(pd: Any, f: Frames) -> Any:
    return f["keyed"].merge(f["lookup"], on="id2", how="left")


@operation("concat two frames", "reshape", "numeric", "pandas.concat")
def _concat(pd: Any, f: Frames) -> Any:
    return pd.concat([f["numeric"], f["numeric"]], ignore_index=True)


@operation("pivot_table", "reshape", "keyed", "DataFrame.pivot_table")
def _pivot(pd: Any, f: Frames) -> Any:
    return f["keyed"].pivot_table(index="id1", columns="v1", values="v2", aggfunc="mean")


@operation("drop_duplicates", "basics", "keyed", "DataFrame.drop_duplicates")
def _drop_duplicates(pd: Any, f: Frames) -> Any:
    return f["keyed"].drop_duplicates(subset=["id1", "id2"])


@operation("set_index then loc", "indexing", "keyed", "DataFrame.set_index DataFrame.loc")
def _set_index(pd: Any, f: Frames) -> Any:
    return f["keyed"].set_index("id3").sort_index().loc[:1000]


@operation("dt.year", "temporal", "temporal", "dt.year")
def _dt_year(pd: Any, f: Frames) -> Any:
    return f["temporal"]["t"].dt.year


@operation("dt.floor to the hour", "temporal", "temporal", "dt.floor")
def _dt_floor(pd: Any, f: Frames) -> Any:
    return f["temporal"]["t"].dt.floor("h")


@operation("resample.sum hourly", "groupby", "temporal", "DataFrame.resample Resampler.sum")
def _resample(pd: Any, f: Frames) -> Any:
    return f["temporal"].set_index("t")["v"].resample("h").sum()


@operation("astype to category", "categorical", "keyed", "DataFrame.astype")
def _categorical(pd: Any, f: Frames) -> Any:
    return f["keyed"]["id1"].astype("category")


# The eleven below come from firepanda-bench rather than from the parity sections. Every
# one of them is an operation a published benchmark query actually runs and this matrix
# had no row for, which is a better filter than picking operations that look important.
# The list came out of linking each bench query to its operations, and it produced work
# rather than opinions.


@operation("groupby.median", "groupby", "keyed", "DataFrame.groupby GroupBy.median")
def _groupby_median(pd: Any, f: Frames) -> Any:
    # db-benchmark q6, and the most interesting row added here. A median cannot be
    # computed with a running accumulator, so the values have to be kept per group,
    # and until this row existed the matrix measured no reduction that needs per group
    # memory at all. That is the thing the memory half of the table exists to expose.
    return f["keyed"].groupby("id2")["v2"].median()


@operation("groupby.std", "groupby", "keyed", "GroupBy.std")
def _groupby_std(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id2")["v2"].std()


@operation("groupby.min", "groupby", "keyed", "GroupBy.min")
def _groupby_min(pd: Any, f: Frames) -> Any:
    # `GroupBy.max` had a row and its opposite did not, which was an accident rather
    # than a decision. They are not always the same code path.
    return f["keyed"].groupby("id2")["v1"].min()


@operation("groupby.count", "groupby", "keyed", "GroupBy.count")
def _groupby_count(pd: Any, f: Frames) -> Any:
    # Counting non null values, which is not the same as `size` and is the one TPC-H
    # q13 calls.
    return f["keyed"].groupby("id2")["v1"].count()


@operation("groupby.head 2 per group", "groupby", "keyed", "GroupBy.head")
def _groupby_head(pd: Any, f: Frames) -> Any:
    # db-benchmark q8, which is the query that separates an engine with a real group by
    # from one faking it with a sort. An order statistic per group rather than a
    # reduction, and the answer is large where every other grouped row here is small.
    return f["keyed"].groupby("id2").head(2)


@operation("str.startswith", "strings", "strings", "str.startswith")
def _startswith(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.startswith("ab")


@operation("str.endswith", "strings", "strings", "str.endswith")
def _endswith(pd: Any, f: Frames) -> Any:
    # Anchored at the other end, which is a different kernel in an engine that stores
    # offsets, because the length has to be read before the comparison can start.
    return f["strings"]["s"].str.endswith("z")


@operation("str.slice", "strings", "strings", "str.slice")
def _slice(pd: Any, f: Frames) -> Any:
    # TPC-H q22 takes the first two characters of a key column. Positional rather than
    # searching, so an engine that has to decode UTF-8 to find character two pays for
    # it here and an engine that does not, does not.
    return f["strings"]["s"].str.slice(0, 2)


@operation("map through a dict", "basics", "keyed", "Series.map")
def _map(pd: Any, f: Frames) -> Any:
    # TPC-H q2 and q17 both map a key to a per key aggregate through a dict, which is
    # what is measured here rather than `map` with a lambda. The lambda form is the
    # interpreter and would be the largest ratio in the table by a wide margin, and it
    # is also not what the queries do.
    table = {value: float(value) for value in range(GROUPS)}
    return f["keyed"]["id2"].map(table)


@operation("mean", "stats", "numeric", "Series.mean")
def _mean(pd: Any, f: Frames) -> Any:
    return f["numeric"]["c"].mean()


@operation("assign two columns", "basics", "numeric", "DataFrame.assign")
def _assign(pd: Any, f: Frames) -> Any:
    frame = f["numeric"]
    return frame.assign(d=frame["a"] * 2, e=frame["c"] + 1.0)


# The chained cases. These are the rows where a tenth of the memory is actually
# available, because peak memory in pandas is dominated by intermediates and a chain is
# where the intermediates are. A single reduction cannot use much less memory than its
# input however good the engine is, so a memory ratio on one of those says more about
# how the frame is stored than about the engine.


@operation(
    "filter then groupby.sum",
    "groupby",
    "keyed",
    "DataFrame.loc DataFrame.groupby GroupBy.sum",
    chained=True,
)
def _filter_group(pd: Any, f: Frames) -> Any:
    frame = f["keyed"]
    return frame[frame["v1"] > 50].groupby("id2")["v2"].sum()


@operation(
    "merge then groupby.mean",
    "reshape",
    "keyed lookup",
    "DataFrame.merge DataFrame.groupby GroupBy.mean",
    chained=True,
)
def _merge_group(pd: Any, f: Frames) -> Any:
    return f["keyed"].merge(f["lookup"], on="id2", how="left").groupby("id1")["w"].mean()


@operation(
    "cast then sort then head",
    "indexing",
    "numeric",
    "DataFrame.astype DataFrame.sort_values DataFrame.head",
    chained=True,
)
def _cast_sort_head(pd: Any, f: Frames) -> Any:
    frame = f["numeric"]
    return frame.assign(a=frame["a"].astype("float64")).sort_values("a").head(1000)


@operation(
    "groupby then sort then head",
    "groupby",
    "keyed",
    "DataFrame.groupby GroupBy.sum DataFrame.sort_values",
    chained=True,
)
def _group_sort_head(pd: Any, f: Frames) -> Any:
    return f["keyed"].groupby("id2")["v1"].sum().sort_values(ascending=False).head(10)


@operation(
    "filter then sort then groupby",
    "indexing",
    "keyed",
    "DataFrame.loc DataFrame.sort_values DataFrame.groupby",
    chained=True,
)
def _filter_sort_group(pd: Any, f: Frames) -> Any:
    frame = f["keyed"]
    return frame[frame["v2"] > 0.5].sort_values("id3").groupby("id1")["v1"].last()


@operation(
    "str.contains then groupby.size",
    "strings",
    "strings",
    "str.contains DataFrame.groupby GroupBy.size",
    chained=True,
)
def _contains_group(pd: Any, f: Frames) -> Any:
    frame = f["strings"]
    return frame[frame["s"].str.contains("a", regex=False)].groupby("key").size()


@operation(
    "str.upper then drop_duplicates",
    "strings",
    "strings",
    "str.upper Series.drop_duplicates",
    chained=True,
)
def _upper_dedupe(pd: Any, f: Frames) -> Any:
    return f["strings"]["s"].str.upper().drop_duplicates()


@operation(
    "floor then resample then cumsum",
    "temporal",
    "temporal",
    "dt.floor DataFrame.resample Series.cumsum",
    chained=True,
)
def _floor_resample_cumsum(pd: Any, f: Frames) -> Any:
    frame = f["temporal"]
    hourly = frame.assign(t=frame["t"].dt.floor("h")).set_index("t")["v"]
    return hourly.resample("h").sum().cumsum()


@operation(
    "value_counts then head", "stats", "keyed", "Series.value_counts DataFrame.head", chained=True
)
def _counts_head(pd: Any, f: Frames) -> Any:
    return f["keyed"]["id3"].value_counts().head(20)


@operation(
    "drop_duplicates then merge",
    "reshape",
    "keyed lookup",
    "DataFrame.drop_duplicates DataFrame.merge",
    chained=True,
)
def _dedupe_merge(pd: Any, f: Frames) -> Any:
    return f["keyed"].drop_duplicates(subset=["id2"]).merge(f["lookup"], on="id2", how="inner")


@operation(
    "cast then groupby.agg",
    "basics",
    "keyed",
    "DataFrame.astype DataFrame.groupby GroupBy.agg",
    chained=True,
)
def _cast_group_agg(pd: Any, f: Frames) -> Any:
    frame = f["keyed"]
    return frame.assign(v1=frame["v1"].astype("float64")).groupby("id2").agg({"v1": "mean"})


@operation(
    "concat then sort then drop_duplicates",
    "reshape",
    "numeric",
    "pandas.concat DataFrame.sort_values DataFrame.drop_duplicates",
    chained=True,
)
def _concat_sort_dedupe(pd: Any, f: Frames) -> Any:
    both = pd.concat([f["numeric"], f["numeric"]], ignore_index=True)
    return both.sort_values("a").drop_duplicates(subset=["a", "b"])


@operation(
    "merge then filter then sort",
    "reshape",
    "keyed lookup",
    "DataFrame.merge DataFrame.loc DataFrame.sort_values",
    chained=True,
)
def _merge_filter_sort(pd: Any, f: Frames) -> Any:
    joined = f["keyed"].merge(f["lookup"], on="id2", how="inner")
    return joined[joined["w"] > 0].sort_values("w")


@operation(
    "groupby.agg then merge",
    "reshape",
    "keyed lookup",
    "GroupBy.agg DataFrame.reset_index DataFrame.merge",
    chained=True,
)
def _agg_merge(pd: Any, f: Frames) -> Any:
    totals = f["keyed"].groupby("id2").agg({"v1": "sum"}).reset_index()
    return totals.merge(f["lookup"], on="id2", how="left")


@operation(
    "category then groupby.sum",
    "categorical",
    "keyed",
    "DataFrame.astype DataFrame.groupby GroupBy.sum",
    chained=True,
)
def _category_group(pd: Any, f: Frames) -> Any:
    # The classic memory win a user is told to reach for, so it is worth having a row
    # that says how much of one it actually is in each engine.
    frame = f["keyed"]
    return (
        frame.assign(id1=frame["id1"].astype("category")).groupby("id1", observed=True)["v1"].sum()
    )


@operation(
    "str.len then filter then mean",
    "strings",
    "strings",
    "str.len DataFrame.loc GroupBy.mean",
    chained=True,
)
def _len_filter_mean(pd: Any, f: Frames) -> Any:
    frame = f["strings"]
    long_enough = frame[frame["s"].str.len() > 12]
    return long_enough.groupby("key")["key"].mean()


@operation(
    "rolling then dropna then max",
    "windows",
    "numeric",
    "Rolling.mean Series.dropna Series.max",
    chained=True,
)
def _rolling_dropna_max(pd: Any, f: Frames) -> Any:
    return f["numeric"]["c"].rolling(100).mean().dropna().max()


@operation(
    "rank then filter then groupby.max",
    "stats",
    "keyed",
    "Series.rank DataFrame.loc GroupBy.max",
    chained=True,
)
def _rank_filter_group(pd: Any, f: Frames) -> Any:
    frame = f["keyed"]
    ranked = frame.assign(r=frame["v2"].rank())
    return ranked[ranked["r"] > len(ranked) / 2].groupby("id1")["r"].max()


@operation(
    "dt.year then groupby.mean",
    "temporal",
    "temporal",
    "dt.year DataFrame.groupby GroupBy.mean",
    chained=True,
)
def _year_group(pd: Any, f: Frames) -> Any:
    frame = f["temporal"]
    return frame.assign(year=frame["t"].dt.year).groupby("year")["v"].mean()


@operation(
    "isin then groupby.nunique",
    "basics",
    "keyed",
    "Series.isin DataFrame.loc GroupBy.nunique",
    chained=True,
)
def _isin_group_nunique(pd: Any, f: Frames) -> Any:
    frame = f["keyed"]
    return frame[frame["v1"].isin(list(range(50)))].groupby("id1")["id3"].nunique()


@operation(
    "filter then groupby.median",
    "groupby",
    "keyed",
    "DataFrame.loc DataFrame.groupby GroupBy.median",
    chained=True,
)
def _filter_group_median(pd: Any, f: Frames) -> Any:
    # The chained twin of `groupby.median`, and the reason it is here rather than the
    # plain row being enough. A median keeps the values per group, a filter in front of
    # it changes how many values that is, and the peak of the pair is where an engine
    # that streams the filter into the aggregation separates from one that materializes
    # the filtered frame first. Neither of those shows up in the single row.
    frame = f["keyed"]
    return frame[frame["v1"] > 50].groupby("id2")["v2"].median()


def registry() -> dict[str, Operation]:
    """Every operation by id.

    Returns:
        Operation id to operation.

    Raises:
        ValueError: When two operations share an id, since the matrix is keyed by it
            and the second one would silently replace the first.
    """
    found: dict[str, Operation] = {}
    for item in OPERATIONS:
        if item.id in found:
            raise ValueError(f"two operations are called {item.id!r}")
        found[item.id] = item
    return found


def described(items: Iterable[Operation]) -> dict[str, Any]:
    """The part of an operation that is worth reading without running anything.

    Args:
        items: The operations to describe.

    Returns:
        Operation id to its section, the pandas names it covers, whether it is a
        chain, and which frames it needs.
    """
    return {
        item.id: {
            "section": item.section,
            "covers": list(item.covers),
            "chained": item.chained,
            "needs": list(item.needs),
        }
        for item in items
    }


def publish(check: bool) -> int:
    """Writes the operation table, or checks that the committed one is current.

    The matrix is measured here and read elsewhere. firepanda-bench wants to say which
    operations a benchmark query is made of, and it cannot import this package, because
    a bench environment carries Polars, DuckDB and cuDF and a conformance environment
    deliberately carries none of them. So the table is committed as a file, the same way
    the pandas inventory is, and the other repository vendors a copy of it. A committed
    file that CI checks is the difference between a link that goes stale quietly and one
    that fails a build.

    What is deliberately not in here is any number. Timings belong to a machine and this
    file belongs to a commit, so publishing a median here would produce a number people
    quote without the machine attached.

    Args:
        check: Verify rather than write.

    Returns:
        A process exit status.
    """
    registry()
    document = {
        "generator": "fpcompat.budget",
        "count": len(OPERATIONS),
        "chained": sum(1 for item in OPERATIONS if item.chained),
        "operations": described(OPERATIONS),
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if not check:
        TABLE.write_text(rendered)
        print(f"wrote {TABLE.name}, {len(OPERATIONS)} operations")
        return 0
    if not TABLE.exists():
        print(f"no {TABLE.name}. Run `pixi run operations`", file=sys.stderr)
        return 1
    if TABLE.read_text() != rendered:
        print(
            f"{TABLE.name} is not what the registry says. "
            f"Run `pixi run operations` and commit the diff",
            file=sys.stderr,
        )
        return 1
    print(f"{TABLE.name} is current, {len(OPERATIONS)} operations")
    return 0


# ---------------------------------------------------------------------------
# The measurement, which happens in the child
# ---------------------------------------------------------------------------


def threads() -> int:
    """How many threads this process has right now.

    Only Linux answers this cheaply, through `/proc/self/status`. macOS has no
    equivalent that does not cost a library, so this returns zero there and the result
    file says zero rather than guessing. A wrong thread count is worse than an absent
    one, because a four times speedup on sixteen threads and one on a single thread are
    not the same result and this is the number that tells them apart.

    Returns:
        The thread count, or zero when the platform will not say.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def usage() -> dict[str, float]:
    """This process's peak memory, CPU time and page faults.

    Returns:
        The counters, with memory in bytes on both platforms.
    """
    used = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_bytes": used.ru_maxrss * RSS_UNIT,
        "cpu_user_s": used.ru_utime,
        "cpu_sys_s": used.ru_stime,
        "minor_faults": used.ru_minflt,
        "major_faults": used.ru_majflt,
    }


def consume(answer: Any) -> int:
    """Forces an answer to exist, and returns something cheap about it.

    A timer around a call that returns a lazy handle measures how fast an engine can
    promise to do the work. firepanda is lazy underneath after M4, so every operation
    here would read as instant without this.

    The order is deliberate. `collect` and `to_arrow` are the two spellings a lazy
    frame uses to say "actually do it", and either is an honest force. Failing those,
    reading the shape and touching one value is the cheapest thing a reasonable
    implementation cannot defer. An engine still lazy after all of that is being
    measured wrong, and the fix belongs in that engine's adapter rather than here.

    Args:
        answer: Whatever the operation returned.

    Returns:
        A number derived from the answer, so nothing can optimise the call away.
    """
    for force in ("collect", "to_arrow"):
        method = getattr(answer, force, None)
        if callable(method):
            answer = method()
            break
    if hasattr(answer, "shape"):
        shape = answer.shape
        size = int(shape[0]) if shape else 1
        if size and hasattr(answer, "iloc"):
            answer.iloc[0]
        return size
    if hasattr(answer, "__len__"):
        return len(answer)
    return 1


def iqr(ordered: list[float]) -> float:
    """The interquartile range of a sorted sample.

    Args:
        ordered: The samples, sorted.

    Returns:
        The range, or zero when there are fewer than four samples, which is too few
        for a quartile to mean anything.
    """
    if len(ordered) < 4:
        return 0.0
    half = len(ordered) // 2
    return statistics.median(ordered[-half:]) - statistics.median(ordered[:half])


def measure(module: Any, item: Operation, loaded: Frames, repeats: int) -> dict[str, Any]:
    """Runs one operation `repeats` times and reports what it cost.

    Args:
        module: The engine module.
        item: The operation.
        loaded: The frames it needs, already materialized.
        repeats: How many times to run it.

    Returns:
        The measurement, or an entry with `ok` false and the reason it did not run.
    """
    before = usage()
    samples: list[float] = []
    rows = 0
    peak_threads = 0
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            rows = consume(item.call(module, loaded))
            samples.append(time.perf_counter() - started)
            peak_threads = max(peak_threads, threads())
    except Exception as error:  # noqa: BLE001 - the reason is the result
        return {
            "ok": False,
            "reason": f"{type(error).__name__}: {error}",
            "unimplemented": isinstance(error, (AttributeError, NotImplementedError)),
        }

    after = usage()
    ordered = sorted(samples)
    return {
        "ok": True,
        "rows": rows,
        "repeats": repeats,
        "median_s": statistics.median(ordered),
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "iqr_s": iqr(ordered),
        "peak_rss_bytes": int(after["peak_rss_bytes"]),
        # How far the high water mark moved while the operation ran. The peak on its
        # own includes the input, which every engine pays for and which says more about
        # how the frame is stored than about the operation. This is what the operation
        # itself asked for on top of that, and on a chained row it is the intermediates,
        # which is the number the memory goal is really about.
        "rss_delta_bytes": int(max(0, after["peak_rss_bytes"] - before["peak_rss_bytes"])),
        "cpu_user_s": after["cpu_user_s"] - before["cpu_user_s"],
        "cpu_sys_s": after["cpu_sys_s"] - before["cpu_sys_s"],
        "minor_faults": int(after["minor_faults"] - before["minor_faults"]),
        "major_faults": int(after["major_faults"] - before["major_faults"]),
        "threads_peak": peak_threads,
    }


def worker(engine: str, operation_id: str, rows: int, repeats: int) -> int:
    """Measures one operation in this process and prints the result as JSON.

    This is the child, and it exists so that the peak resident set it reports belongs
    to this operation and not to the one before it. That is the whole reason the matrix
    cannot be a loop in one interpreter.

    Args:
        engine: Which engine to load.
        operation_id: Which operation to run.
        rows: The corpus size.
        repeats: How many times to run it.

    Returns:
        A process exit status.
    """
    from fpcompat.engines import load

    if not ready(rows):
        # Deliberately not built here. Generating a million rows of strings inside the
        # process being measured puts the generator's own high water mark into the
        # operation's peak, which is a memory column that is partly about numpy.
        print(
            f"no budget corpus at {rows} rows. Run `pixi run budget-corpus --rows {rows}`",
            file=sys.stderr,
        )
        return 1

    item = registry()[operation_id]
    subject = load(engine)
    module = subject.module()
    loaded = {name: subject.frame(frame_name(rows, name)) for name in item.needs}

    record = measure(module, item, loaded, repeats)
    record["id"] = operation_id
    print(json.dumps(record), flush=True)
    return 0


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------


def machine() -> dict[str, Any]:
    """What the numbers were measured on.

    A timing without a machine is not reproducible, and a memory number without one is
    not even comparable, because the peak includes an allocator that behaves
    differently under a different page size.

    Returns:
        The machine description.
    """
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cores": os.cpu_count() or 0,
        "python": platform.python_version(),
    }


def child(engine: str, operation_id: str, rows: int, repeats: int) -> dict[str, Any]:
    """Runs one operation in its own process and parses what it printed.

    A child that dies is a result and not an exception. An operation that segfaults an
    engine is exactly the kind of thing this table exists to publish, and losing the
    whole sweep to it would mean nobody ever sees the row.

    Args:
        engine: Which engine.
        operation_id: Which operation.
        rows: The corpus size.
        repeats: Repeats.

    Returns:
        The measurement, or a failure entry carrying whatever the child said.
    """
    command = [
        sys.executable,
        "-m",
        "fpcompat.budget",
        "--worker",
        "--engine",
        engine,
        "--operation",
        operation_id,
        "--rows",
        str(rows),
        "--repeats",
        str(repeats),
    ]
    finished = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, check=False)
    lines = finished.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        detail = finished.stderr.strip().splitlines()[-1] if finished.stderr.strip() else ""
        return {
            "ok": False,
            "reason": f"the worker exited {finished.returncode}: {detail or 'no output'}",
            "unimplemented": False,
        }


def sweep(engine: str, rows: int, repeats: int, only: list[str] | None = None) -> dict[str, Any]:
    """Runs every operation, one process each.

    Args:
        engine: Which engine to measure.
        rows: The corpus size.
        repeats: Repeats per operation.
        only: Operation ids to run, or None for all of them.

    Returns:
        The result document.
    """
    items = [item for item in OPERATIONS if not only or item.id in only]
    started = time.perf_counter()
    records: dict[str, Any] = {}
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {engine} {item.id}", file=sys.stderr, flush=True)
        records[item.id] = child(engine, item.id, rows, repeats)
    return {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine": engine,
        "rows": rows,
        "repeats": repeats,
        "machine": machine(),
        "seconds": round(time.perf_counter() - started, 3),
        # The operation table travels with the results, for the same reason the
        # conformance results carry their declarations. A result file that needs this
        # repository at the matching commit to be readable is not a result file, it is
        # a cache.
        "operations": described(items),
        "records": records,
    }


def result_path(engine: str, rows: int) -> Path:
    """Where a sweep is written.

    The size is in the name, because a matrix that put a one million row pandas run
    next to a ten million row firepanda run would produce a ten times speedup out of
    nothing and it would look exactly like a real one.

    Args:
        engine: Which engine.
        rows: The corpus size.

    Returns:
        The path.
    """
    return RESULTS / f"budget-{engine}-{rows}.json"


# ---------------------------------------------------------------------------
# The baseline and the gate
# ---------------------------------------------------------------------------

# How much slower a row is allowed to get before the gate says so. Ten percent is
# above the run to run spread we see on a quiet machine and below anything that would
# be called a regression in a review.
SLACK = 0.10


def machine_key() -> str:
    """A short name for this machine, for the baseline filename.

    Not the hostname, which is somebody's laptop name and does not belong in a public
    repository. The system, the architecture and the core count are what actually make
    two timings comparable, and a baseline recorded on a four core runner is not a
    baseline for a sixteen core desktop whatever else they have in common. The full
    machine description goes inside the file, so a run against a baseline that happens
    to share this key but not the processor can still be spotted.

    Returns:
        The key.
    """
    return f"{platform.system().lower()}-{platform.machine()}-{os.cpu_count() or 0}core"


def baseline_path(engine: str, rows: int, key: str | None = None) -> Path:
    """Where a committed baseline lives.

    Args:
        engine: Which engine.
        rows: The corpus size.
        key: The machine key, or None for this machine.

    Returns:
        The path.
    """
    return ROOT / "baselines" / f"{key or machine_key()}-{engine}-{rows}.json"


def baseline_of(document: dict[str, Any]) -> dict[str, Any]:
    """Reduces a sweep to the numbers the gate compares.

    Two per row and nothing else. A baseline that carried the page fault counts would
    be a baseline that changes when the allocator changes, and then it gets
    regenerated on every pull request until nobody reads the diff.

    Args:
        document: A sweep.

    Returns:
        The committed baseline document.
    """
    return {
        "engine": document["engine"],
        "rows": document["rows"],
        "repeats": document["repeats"],
        # Everything about the machine except its name. A hostname is somebody's
        # laptop and this file is public, while the processor and the core count are
        # what actually make two timings comparable.
        "machine": {key: value for key, value in document["machine"].items() if key != "host"},
        "rows_measured": {
            name: {
                "median_s": round(entry["median_s"], 6),
                "peak_rss_bytes": entry["peak_rss_bytes"],
            }
            for name, entry in sorted(document["records"].items())
            if entry.get("ok")
        },
    }


def regressions(document: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Every row that got more than `SLACK` worse than the baseline.

    A row that vanished is a regression too. An operation that used to run and now
    does not is the largest regression available, and a gate that only compares the
    rows both sides have would score it as a clean pass.

    Args:
        document: The sweep just measured.
        baseline: The committed baseline.

    Returns:
        One line per regression, empty when there are none.
    """
    found: list[str] = []
    for name, old in sorted(baseline["rows_measured"].items()):
        entry = document["records"].get(name)
        if not entry or not entry.get("ok"):
            reason = (entry or {}).get("reason", "it was not in this run")
            found.append(f"{name}: ran in the baseline and did not run here, {reason}")
            continue
        for field, unit, was, now in (
            ("time", "ms", old["median_s"], entry["median_s"]),
            ("peak memory", "MB", old["peak_rss_bytes"], entry["peak_rss_bytes"]),
        ):
            if was and now > was * (1 + SLACK):
                render_was = milliseconds(was) if unit == "ms" else megabytes(was)
                render_now = milliseconds(now) if unit == "ms" else megabytes(now)
                found.append(
                    f"{name}: {field} went from {render_was} to {render_now} {unit}, "
                    f"which is {now / was:.2f} times the baseline"
                )
    return found


def gate(engine: str, rows: int) -> int:
    """Checks a sweep against the committed baseline for this machine.

    The gate belongs on a machine that is always the same one. A timing gate on a
    shared CI runner fails on a noisy neighbour and passes on a real regression that
    happened to land on a quiet one, and after the third false alarm somebody adds a
    retry and it stops being a gate at all.

    Args:
        engine: Which engine.
        rows: The corpus size.

    Returns:
        A process exit status.
    """
    result = result_path(engine, rows)
    if not result.exists():
        print(f"no {result.relative_to(ROOT)}. Run `pixi run budget` first")
        return 1
    path = baseline_path(engine, rows)
    if not path.exists():
        print(
            f"no baseline at {path.relative_to(ROOT)}. Run "
            f"`pixi run budget-baseline` on this machine and commit it"
        )
        return 1

    document = json.loads(result.read_text())
    baseline = json.loads(path.read_text())
    if baseline["machine"]["processor"] != document["machine"]["processor"]:
        print(
            f"the baseline was measured on {baseline['machine']['processor']} and this "
            f"run is on {document['machine']['processor']}. Same key, different "
            f"machine, so the comparison would be about the hardware",
            file=sys.stderr,
        )
        return 1

    found = regressions(document, baseline)
    if found:
        print(f"{len(found)} rows regressed by more than {SLACK:.0%}:", file=sys.stderr)
        for line in found:
            print(f"  {line}", file=sys.stderr)
        print(
            "either fix it, or raise the baseline in the same pull request with a line "
            "saying why the row costs more now",
            file=sys.stderr,
        )
        return 1
    print(f"{len(baseline['rows_measured'])} rows within {SLACK:.0%} of {path.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def milliseconds(value: float) -> str:
    """Formats a duration for the table."""
    return f"{value * 1000:.1f}"


def megabytes(value: float) -> str:
    """Formats a byte count for the table."""
    return f"{value / (1 << 20):.0f}"


def median(values: list[float]) -> float:
    """The median, or zero when there is nothing to take one of."""
    return statistics.median(values) if values else 0.0


def one_engine(baseline: dict[str, Any]) -> list[str]:
    """The table with only a baseline in it.

    This exists on purpose rather than as a fallback. A baseline measured after the
    subject is a baseline measured to make the subject look a particular way, so the
    pandas column gets published before there is anything to compare it with.

    Args:
        baseline: The pandas sweep.

    Returns:
        The markdown lines.
    """
    lines = [
        f"One engine here, which is {baseline['engine']}. The comparison columns appear "
        "when there is a second engine that can run these operations.",
        "",
        "| Operation | Section | ms | IQR ms | peak MB | delta MB | rows out |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in OPERATIONS:
        entry = baseline["records"].get(item.id)
        if not entry or not entry.get("ok"):
            continue
        lines.append(
            f"| {item.id} | {item.section} | {milliseconds(entry['median_s'])} | "
            f"{milliseconds(entry['iqr_s'])} | {megabytes(entry['peak_rss_bytes'])} | "
            f"{megabytes(entry['rss_delta_bytes'])} | {entry['rows']:,} |"
        )
    return lines


def two_engines(baseline: dict[str, Any], subject: dict[str, Any]) -> list[str]:
    """The comparison table and the summary against the goal.

    Args:
        baseline: The pandas sweep.
        subject: The other engine's sweep.

    Returns:
        The markdown lines.
    """
    name = subject["engine"]
    lines = [
        "Above one is better, on both ratio columns. A row below one is a performance "
        "bug with a name and an input size, and it goes on the issue list the day it "
        "appears. That is a thing this project can promise, and a uniform ten times is "
        "not.",
        "",
        f"| Operation | Section | pandas ms | {name} ms | speed | pandas MB | {name} MB | memory |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    speeds: list[float] = []
    memories: list[float] = []
    chained_speeds: list[float] = []
    chained_memories: list[float] = []
    for item in OPERATIONS:
        base = baseline["records"].get(item.id)
        mine = subject["records"].get(item.id)
        if not (base and mine and base.get("ok") and mine.get("ok")):
            continue
        speed = base["median_s"] / mine["median_s"] if mine["median_s"] else 0.0
        memory = base["peak_rss_bytes"] / mine["peak_rss_bytes"] if mine["peak_rss_bytes"] else 0.0
        speeds.append(speed)
        memories.append(memory)
        if item.chained:
            chained_speeds.append(speed)
            chained_memories.append(memory)
        lines.append(
            f"| {item.id} | {item.section} | {milliseconds(base['median_s'])} | "
            f"{milliseconds(mine['median_s'])} | {speed:.2f}x | "
            f"{megabytes(base['peak_rss_bytes'])} | {megabytes(mine['peak_rss_bytes'])} | "
            f"{memory:.2f}x |"
        )

    slow = sum(1 for value in speeds if value < 1.0)
    heavy = sum(1 for value in memories if value < 1.0)
    lines += [
        "",
        "## Against the goal",
        "",
        "| | rows | speed | memory |",
        "| --- | ---: | ---: | ---: |",
        f"| every row both engines ran | {len(speeds)} | {median(speeds):.2f}x | "
        f"{median(memories):.2f}x |",
        f"| the chained rows | {len(chained_speeds)} | {median(chained_speeds):.2f}x | "
        f"{median(chained_memories):.2f}x |",
        f"| rows below pandas | | {slow} | {heavy} |",
        "",
        "Medians rather than means, because a hundred times on one operation and a tie "
        "everywhere else is not a hundred times engine, and an arithmetic mean says it "
        "is. The target is ten and ten, and the chained rows are the honest place to "
        "read the memory number.",
    ]
    return lines


def refusals(document: dict[str, Any]) -> list[str]:
    """The rows an engine did not run, and why.

    Dropping them would turn a partial implementation into a clean sweep, which is the
    single easiest way to publish a dishonest table.

    Args:
        document: A sweep.

    Returns:
        The markdown lines, empty when everything ran.
    """
    failed = sorted(
        (name, entry) for name, entry in document["records"].items() if not entry.get("ok")
    )
    if not failed:
        return []
    lines = [
        "",
        f"## {len(failed)} operations {document['engine']} did not run",
        "",
        "Unimplemented is a schedule and an error is a bug, and they are kept apart "
        "here for the same reason the conformance runner keeps them apart.",
        "",
        "| Operation | Kind | Reason |",
        "| --- | --- | --- |",
    ]
    for name, entry in failed:
        kind = "unimplemented" if entry.get("unimplemented") else "error"
        lines.append(f"| {name} | {kind} | {entry.get('reason', '')} |")
    return lines


def matrix(baseline: dict[str, Any], subject: dict[str, Any] | None) -> str:
    """Renders the cost matrix.

    Args:
        baseline: The pandas sweep.
        subject: The other engine's sweep, or None when there is not one yet.

    Returns:
        Markdown.
    """
    where = baseline["machine"]
    lines = [
        f"# Cost matrix at {baseline['rows']:,} rows",
        "",
        f"{where['processor']}, {where['cores']} cores, {where['platform']}. "
        f"{baseline['repeats']} repeats per operation, the median reported and the "
        f"interquartile range beside it. One process per engine per operation, because "
        f"a peak resident set is a property of a process and not of a call.",
        "",
        "Peak is the whole process, which includes the input every engine has to hold. "
        "Delta is how far the high water mark moved while the operation ran, which is "
        "the intermediates.",
        "",
    ]
    lines += one_engine(baseline) if subject is None else two_engines(baseline, subject)
    lines += refusals(baseline)
    if subject is not None:
        lines += refusals(subject)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def build_corpus(rows: int, check: bool) -> int:
    """Builds the budget corpus and reconciles it with the committed manifest.

    Args:
        rows: The corpus size.
        check: Whether a difference is an error rather than something to write down.

    Returns:
        A process exit status.
    """
    built = write(rows)
    entry = describe(rows, built)
    document = read_manifest()
    committed = document["sizes"].get(str(rows))

    if committed and committed != entry:
        for name in sorted(entry["frames"]):
            fresh = entry["frames"][name]
            old = committed.get("frames", {}).get(name, {})
            if old != fresh:
                print(
                    f"{name} at {rows} rows changed: the manifest says "
                    f"{old.get('digest', 'nothing')[:12]}, generated "
                    f"{fresh['digest'][:12]}",
                    file=sys.stderr,
                )
        if check:
            print(
                "run `pixi run budget-corpus` and commit the manifest diff, which is "
                "the record of what changed about the inputs",
                file=sys.stderr,
            )
            return 1

    document["sizes"][str(rows)] = entry
    text_out = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if not check and (not MANIFEST.exists() or MANIFEST.read_text() != text_out):
        MANIFEST.write_text(text_out)
        print(f"wrote {MANIFEST.relative_to(ROOT)}")
    total = sum(table.num_rows for table in built.values())
    print(f"{len(built)} frames, {total:,} rows, under {(BUDGET / str(rows)).relative_to(ROOT)}")
    return 0


def render(engine: str, against: str, rows: int, out: Path | None) -> int:
    """Renders the matrix from the result files.

    Args:
        engine: The subject engine.
        against: The baseline engine.
        rows: The corpus size.
        out: Where to write it, or None to print it.

    Returns:
        A process exit status.
    """
    baseline_path = result_path(against, rows)
    if not baseline_path.exists():
        print(f"no {baseline_path.relative_to(ROOT)}. Run `pixi run budget` first")
        return 1
    baseline = json.loads(baseline_path.read_text())

    subject = None
    if engine != against:
        subject_path = result_path(engine, rows)
        if not subject_path.exists():
            print(
                f"no {subject_path.relative_to(ROOT)}. Run "
                f"`pixi run budget --engine {engine}` first"
            )
            return 1
        subject = json.loads(subject_path.read_text())

    document = matrix(baseline, subject)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document)
        print(f"wrote {out}")
    else:
        print(document, end="")
    return 0


def record_baseline(engine: str, rows: int) -> int:
    """Writes this machine's baseline from the last sweep.

    Args:
        engine: Which engine.
        rows: The corpus size.

    Returns:
        A process exit status.
    """
    result = result_path(engine, rows)
    if not result.exists():
        print(f"no {result.relative_to(ROOT)}. Run `pixi run budget` first")
        return 1
    document = json.loads(result.read_text())
    path = baseline_path(engine, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline_of(document), indent=2, sort_keys=True) + "\n")
    print(
        f"{len(baseline_of(document)['rows_measured'])} rows, wrote "
        f"{path.relative_to(ROOT)}. Commit it, and say in the pull request what "
        f"machine it is"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Builds the corpus, runs a sweep or renders the matrix.

    Args:
        argv: Command line arguments.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description="The operation level cost matrix.")
    parser.add_argument("--engine", default="pandas", help="which engine to measure")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="corpus size")
    parser.add_argument("--large", action="store_true", help=f"{LARGE_ROWS:,} rows")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="runs per operation")
    parser.add_argument("--only", action="append", help="run one operation, repeatable")
    parser.add_argument("--corpus", action="store_true", help="build the budget corpus")
    parser.add_argument("--check", action="store_true", help="verify the corpus digests")
    parser.add_argument("--matrix", action="store_true", help="render from the result files")
    parser.add_argument("--against", default="pandas", help="the baseline engine")
    parser.add_argument("--out", type=Path, help="write the matrix here")
    parser.add_argument("--list", action="store_true", help="print the operation ids")
    parser.add_argument(
        "--operations", action="store_true", help="write the committed operation table"
    )
    parser.add_argument("--baseline", action="store_true", help="record this machine's baseline")
    parser.add_argument("--gate", action="store_true", help="fail on a regression past the slack")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operation", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    rows = LARGE_ROWS if args.large else args.rows
    registry()

    if args.list:
        for item in OPERATIONS:
            print(item.id)
        return 0

    if args.operations:
        return publish(args.check)

    if args.corpus or args.check:
        return build_corpus(rows, args.check)

    if args.worker:
        return worker(args.engine, args.operation, rows, args.repeats)

    if args.matrix:
        return render(args.engine, args.against, rows, args.out)

    if args.gate:
        return gate(args.engine, rows)

    if args.baseline:
        return record_baseline(args.engine, rows)

    if not ready(rows):
        print(f"building the budget corpus at {rows:,} rows", file=sys.stderr)
        write(rows)

    document = sweep(args.engine, rows, args.repeats, args.only)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = result_path(args.engine, rows)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    ran = sum(1 for entry in document["records"].values() if entry.get("ok"))
    print(
        f"{ran} of {len(document['records'])} operations in {document['seconds']}s, "
        f"wrote {path.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
