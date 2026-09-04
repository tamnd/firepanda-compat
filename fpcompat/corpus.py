"""The frames every case runs on.

Implements `docs/specs/04-corpus.md`.

Small and mean rather than large and representative. Ten million rows of well
formed float64 exercises the same three branches as a thousand rows of it. What
finds bugs is a column that is all null, a group with one member, a string of
exactly twelve bytes, a negative zero sitting next to a NaN, and a timestamp on
the day Lord Howe Island moves its clocks by half an hour. Every frame here is
between 0 and 10000 rows, the whole corpus regenerates in about a second, and size
belongs in firepanda-bench where it is the point.

Three things about this module are deliberate.

The random part is splitmix64 in its counter form, with the same constants and the
same seed as `tools/data.py` in firepanda-bench, which is the hexadecimal digits
of pi. Same generator, same seed, same stream, so the Mojo driver can produce a
column without reading a file and a person reading both repositories does not have
to wonder whether the difference between them means something.

The interesting part is not random at all. Negative zero, the smallest denormal
and the largest finite float do not come out of a random generator, so they are
written down as literals at fixed offsets and the manifest records the offsets. A
corpus that waits for a random stream to produce a negative zero is a corpus that
finds those bugs on a Tuesday in 2027.

The manifest is committed and the frames are not. The frames regenerate in a
second and a binary in git is a binary in git forever, while the manifest is what
makes a corpus change reviewable: adding a frame or moving a null shows up as a
diff in the pull request, so a conformance number that moved because the inputs
moved is visible rather than mysterious.

The frames go to Arrow IPC and not to Parquet, which is a change from what document
04 first said and is here because the first version wrote Parquet and the round trip
test caught it. Parquet promotes `timestamp[s]` to `timestamp[ms]`, narrows the
`large_string` inside a dictionary to `string`, and renames the child field of a
large list. Every one of those is a distinction this corpus exists to make: the
resolution frame came back with two millisecond columns where it was written with a
second column and a millisecond one, so a case about non-nanosecond resolution would
have quietly tested nothing. Arrow IPC round trips all 56 frames to the same digest.
CSV is not a corpus format for the same reason and a worse one, since it cannot tell
a null from an empty string; the reader cases will write their own text fixtures,
where the text is the input under test rather than a transport.

Usage:
    python -m fpcompat.corpus            # regenerate the frames and the manifest
    python -m fpcompat.corpus --check    # regenerate and verify every digest
    python -m fpcompat.corpus --list     # the frame names, one per line
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
MANIFEST = CORPUS / "manifest.json"

# The first sixteen hexadecimal digits of pi, which is the seed firepanda-bench
# already uses. The two are deliberately the same constant.
SEED = 0x243F6A8885A308D3

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX_A = np.uint64(0xBF58476D1CE4E5B9)
MIX_B = np.uint64(0x94D049BB133111EB)

# Frames are this many rows unless they exist in order to be a different size.
# Sixty four is enough to hold every edge value at its own offset and few enough
# that a failing case can print the whole column into the report.
ROWS = 64
TALL = 10_000
WIDE_COLUMNS = 200


def splitmix64(seed: int, count: int, skip: int = 0) -> np.ndarray:
    """Returns `count` words of the splitmix64 stream for a seed.

    The counter form, which produces the same sequence as calling `next_u64`
    repeatedly but computes any slice of the stream without computing the slices
    before it. That is what lets a driver generate the second half of a column
    without generating the first, and it is copied from `tools/data.py` in
    firepanda-bench rather than reinvented, because two implementations of one
    generator is two chances to disagree.

    Args:
        seed: The generator seed.
        count: How many words to produce.
        skip: How many words of the stream to pass over first.

    Returns:
        An array of unsigned 64-bit words.
    """
    with np.errstate(over="ignore"):
        steps = np.arange(skip + 1, skip + count + 1, dtype=np.uint64)
        z = np.uint64(seed) + steps * GOLDEN
        z = (z ^ (z >> np.uint64(30))) * MIX_A
        z = (z ^ (z >> np.uint64(27))) * MIX_B
        return z ^ (z >> np.uint64(31))


def below(words: np.ndarray, bound: int) -> np.ndarray:
    """Reduces random words to `[0, bound)`.

    A remainder, matching `Rng.next_below` in the library and `_below` in the
    bench, bias and all. The bias is around two to the minus forty for these
    bounds and it does not move an answer.

    Args:
        words: The random words.
        bound: The exclusive upper bound.

    Returns:
        Values in the half-open range, as int64.
    """
    return (words % np.uint64(bound)).astype(np.int64)


def salt(name: str) -> int:
    """Returns a stable 64-bit salt for a name.

    Python's `hash` is salted per process, so it cannot be used for anything that
    has to reproduce. This is the first eight bytes of the SHA-256 of the name,
    which is stable across processes, machines and languages, and it is what gives
    every frame its own part of the stream.

    Args:
        name: The frame or column name.

    Returns:
        A 64-bit integer.
    """
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def nulls_at(values: list[Any], offsets: list[int]) -> list[Any]:
    """Returns the values with `None` at the given offsets.

    Args:
        values: The values.
        offsets: Row offsets to null out.

    Returns:
        A new list.
    """
    out = list(values)
    for offset in offsets:
        if 0 <= offset < len(out):
            out[offset] = None
    return out


# The six float values that break naive code, at fixed offsets so a failure report
# can say which one it was. The largest finite value is here because a sum over a
# column containing it either overflows to infinity or does not, and those two
# answers look identical on random data.
FLOAT_EDGES: dict[str, float] = {
    "nan": float("nan"),
    "positive_infinity": float("inf"),
    "negative_infinity": float("-inf"),
    "negative_zero": -0.0,
    "smallest_denormal": 5e-324,
    "largest_finite": 1.7976931348623157e308,
}

INT_WIDTHS: dict[str, pa.DataType] = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
}

# Int128 is not here. firepanda has it and pandas does not, so pandas cannot hold
# the frame and cannot be the oracle for it. A dtype the reference implementation
# has no answer for belongs in the library's own tests, and putting it here would
# mean either a case with nothing to compare against or a divergence entry that
# excuses a whole dtype.

FLOAT_WIDTHS: dict[str, pa.DataType] = {"float32": pa.float32(), "float64": pa.float64()}

NULL_SHAPES = ("no_nulls", "half_null", "all_null")


def null_offsets(shape: str, rows: int) -> list[int]:
    """Returns the null row offsets for a null shape.

    Args:
        shape: One of `no_nulls`, `half_null`, `all_null`.
        rows: The row count.

    Returns:
        The offsets.
    """
    if shape == "no_nulls":
        return []
    if shape == "all_null":
        return list(range(rows))
    # Every other row rather than the first half, because a half null column whose
    # nulls are contiguous passes a validity implementation that only ever reads
    # the first word of the bitmap.
    return list(range(1, rows, 2))


def integer_frame(name: str, dtype: pa.DataType, shape: str) -> pa.Table:
    """Builds one integer frame in one null shape.

    Args:
        name: The dtype name.
        dtype: The Arrow type.
        shape: The null shape.

    Returns:
        A table with a positional `row` column and a `value` column.
    """
    bits = dtype.bit_width
    signed = pa.types.is_signed_integer(dtype)
    span = (1 << (bits - 1)) if signed else (1 << bits)
    values = below(splitmix64(SEED ^ salt(name), ROWS), min(span, 1 << 32)).tolist()
    if signed:
        values = [value - (span >> 1) for value in values]
    return pa.table(
        {
            "row": pa.array(range(ROWS), type=pa.int64()),
            "value": pa.array(nulls_at(values, null_offsets(shape, ROWS)), type=dtype),
        }
    )


def float_frame(name: str, dtype: pa.DataType, shape: str) -> pa.Table:
    """Builds one float frame in one null shape, with the six edges in it.

    The edges go at offsets 0 through 5 of every float frame, in the order they
    appear in `FLOAT_EDGES`, so a case that fails at offset 3 failed on a negative
    zero in whichever frame it failed on.

    Args:
        name: The dtype name.
        dtype: The Arrow type.
        shape: The null shape.

    Returns:
        A table with a positional `row` column and a `value` column.
    """
    words = splitmix64(SEED ^ salt(name), ROWS)
    values: list[float | None] = (words >> np.uint64(11)).astype(np.float64).tolist()
    for offset, edge in enumerate(FLOAT_EDGES.values()):
        # float32 cannot hold the float64 denormal or the float64 maximum, and the
        # cast produces zero and infinity rather than an error. That is the right
        # answer and it is what pandas does, so the frame keeps the literal and the
        # narrowing is part of what the case is testing.
        values[offset] = edge
    return pa.table(
        {
            "row": pa.array(range(ROWS), type=pa.int64()),
            "value": pa.array(nulls_at(values, null_offsets(shape, ROWS)), type=dtype),
        }
    )


def shape_frames() -> dict[str, pa.Table]:
    """The frames that exist because of their shape rather than their contents."""
    frames: dict[str, pa.Table] = {}
    schema = pa.schema([("a", pa.int64()), ("b", pa.float64()), ("c", pa.large_string())])

    frames["empty"] = pa.table(
        {
            "a": pa.array([], type=pa.int64()),
            "b": pa.array([], type=pa.float64()),
            "c": pa.array([], type=pa.large_string()),
        },
        schema=schema,
    )
    frames["single"] = pa.table({"a": [1], "b": [1.5], "c": ["one"]}, schema=schema)
    # Two rows, because the off by one in a diff, a shift or a pct_change needs
    # exactly two and is invisible at one and at three.
    frames["two"] = pa.table({"a": [1, 2], "b": [1.5, -0.5], "c": ["one", None]}, schema=schema)

    # Wide, where a column at a time implementation gets slow and where an axis
    # argument gets confused about which way it is pointing.
    wide: dict[str, Any] = {}
    for column in range(WIDE_COLUMNS):
        wide[f"c{column:03d}"] = below(splitmix64(SEED ^ 0x11, 5, skip=column * 5), 1000).tolist()
    frames["wide"] = pa.table(wide)

    frames["tall"] = pa.table(
        {
            "key": below(splitmix64(SEED ^ 0x22, TALL), 100).tolist(),
            "value": (splitmix64(SEED ^ 0x23, TALL) >> np.uint64(11)).astype(np.float64).tolist(),
            "flag": (below(splitmix64(SEED ^ 0x24, TALL), 2) == 1).tolist(),
        }
    )
    return frames


def numeric_frames() -> dict[str, pa.Table]:
    """One frame per numeric width per null shape, plus the integer edges."""
    frames: dict[str, pa.Table] = {}
    for name, dtype in INT_WIDTHS.items():
        for shape in NULL_SHAPES:
            frames[f"{name}_{shape}"] = integer_frame(name, dtype, shape)
    for name, dtype in FLOAT_WIDTHS.items():
        for shape in NULL_SHAPES:
            frames[f"{name}_{shape}"] = float_frame(name, dtype, shape)

    # The minimum and the maximum of every width in the same column as ordinary
    # values, because a sum that promotes correctly and a sum that overflows look
    # identical on random data and differ here on row zero.
    edges: dict[str, Any] = {}
    for name, dtype in INT_WIDTHS.items():
        bits = dtype.bit_width
        if pa.types.is_signed_integer(dtype):
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        else:
            low, high = 0, (1 << bits) - 1
        column = [low, high, low + 1, high - 1, 0, 1]
        column += below(splitmix64(SEED ^ salt(f"edge_{name}"), ROWS - len(column)), 7).tolist()
        edges[name] = pa.array(column, type=dtype)
    frames["integer_edges"] = pa.table(edges)
    return frames


def string_frames() -> dict[str, pa.Table]:
    """The four string frames."""
    frames: dict[str, pa.Table] = {}

    # Lengths 0 through 20, straddling the twelve byte StringView discriminant on
    # purpose. Twelve and thirteen are the two rows worth staring at when a string
    # kernel fails.
    ascii_values = ["".join(chr(ord("a") + (i % 26)) for i in range(n)) for n in range(21)]
    frames["strings_ascii"] = pa.table(
        {
            "length": pa.array(range(21), type=pa.int64()),
            "value": pa.array(ascii_values, type=pa.large_string()),
        }
    )

    unicode_values = [
        "café",  # e plus a combining acute, two code points and one grapheme
        "café",  # the same grapheme precomposed, which is a different string
        "مرحبا",  # right to left
        "\U0001f469‍\U0001f469‍\U0001f467",  # one grapheme, several code points
        "İstanbul",  # dotted capital I, whose lower case is two code points
        "ıstanbul",  # dotless i, so lower and upper are not inverses
        "ß",  # sharp s, whose upper case is two characters and is longer
        "ﬁance",  # the fi ligature, which normalizes into two characters
        "你好",  # three bytes per character in UTF-8
        "\U0001d7d8\U0001d7d9",  # mathematical digits, numeric but not decimal
        "",
        " leading and trailing ",
    ]
    frames["strings_unicode"] = pa.table(
        {
            "row": pa.array(range(len(unicode_values)), type=pa.int64()),
            "value": pa.array(unicode_values, type=pa.large_string()),
        }
    )

    # Two thirds null, with the empty string sitting next to the nulls. A library
    # that confuses an empty string with a null fails here and nowhere else.
    heavy: list[str | None] = []
    for i in range(ROWS):
        heavy.append(None if i % 3 else ("" if i % 6 == 0 else f"v{i}"))
    frames["strings_null_heavy"] = pa.table(
        {
            "row": pa.array(range(ROWS), type=pa.int64()),
            "value": pa.array(heavy, type=pa.large_string()),
        }
    )

    pattern_values = [
        "abc123",
        "ABC123",
        "a.b.c",
        "a|b|c",
        "^anchored$",
        "(group)",
        "[bracket]",
        "back\\slash",
        "tab\tseparated",
        "new\nline",
        "null\x00byte",
        "x" * 500,
        "",
        "  spaced  ",
        "éèê",
        "foobar",
        "foofoobar",
        "barfoo",
    ]
    frames["strings_pattern"] = pa.table(
        {
            "row": pa.array(range(len(pattern_values)), type=pa.int64()),
            "value": pa.array(pattern_values, type=pa.large_string()),
        }
    )
    return frames


def categorical_frames() -> dict[str, pa.Table]:
    """Ordered and unordered categoricals, with an unused category and a null.

    The categories are deliberately not in sorted order, because sorting an
    ordered categorical follows the category order rather than the lexical one, and
    a corpus whose categories are already sorted cannot tell the two apart.
    """
    categories = ["medium", "low", "high", "unused"]
    codes = [0, 1, 2, 0, 1, None, 2, 2, 1, 0, None, 1]
    frames: dict[str, pa.Table] = {}
    for name, ordered in (("categorical_unordered", False), ("categorical_ordered", True)):
        column = pa.DictionaryArray.from_arrays(
            pa.array(codes, type=pa.int32()), pa.array(categories, type=pa.large_string())
        )
        frames[name] = pa.table(
            {
                "row": pa.array(range(len(codes)), type=pa.int64()),
                "value": column.cast(pa.dictionary(pa.int32(), pa.large_string(), ordered=ordered)),
            }
        )
    return frames


def around(zone: str, moment: dt.datetime) -> pa.Table:
    """Builds a window of timestamps either side of a moment, naive and zoned.

    Args:
        zone: The IANA zone name.
        moment: The UTC wall clock moment the window is centred on.

    Returns:
        A table with `naive` and `zoned` columns.
    """
    window = [moment + dt.timedelta(minutes=10 * i) for i in range(-6, 6)]
    return pa.table(
        {
            "row": pa.array(range(len(window)), type=pa.int64()),
            "naive": pa.array(window, type=pa.timestamp("us")),
            "zoned": pa.array(window, type=pa.timestamp("us", tz=zone)),
        }
    )


def temporal_frames() -> dict[str, pa.Table]:
    """Timestamps, dates and durations, including the DST transitions."""
    frames: dict[str, pa.Table] = {}

    base = dt.datetime(2024, 3, 10, 1, 30)
    stamps = [base + dt.timedelta(minutes=15 * i) for i in range(ROWS)]
    resolutions: dict[str, Any] = {"row": pa.array(range(ROWS), type=pa.int64())}
    for unit in ("s", "ms", "us", "ns"):
        resolutions[unit] = pa.array(stamps, type=pa.timestamp(unit))
    frames["temporal_resolutions"] = pa.table(resolutions)

    # Spring forward and fall back in a zone that moves by an hour, and Lord Howe
    # Island, which moves by thirty minutes. The half hour one finds every
    # implementation that assumed a whole hour was involved.
    frames["temporal_dst_forward"] = around("America/New_York", dt.datetime(2024, 3, 10, 7))
    frames["temporal_dst_back"] = around("America/New_York", dt.datetime(2024, 11, 3, 6))
    frames["temporal_dst_lord_howe"] = around("Australia/Lord_Howe", dt.datetime(2024, 10, 5, 15))

    # Before the epoch and after 2262, which is where a nanosecond timestamp runs
    # out of range. pandas 3.0 carries the resolution on the dtype, and these rows
    # are how one finds out whether a library still assumes nanoseconds.
    extremes = [
        dt.datetime(1677, 9, 22),
        dt.datetime(1900, 1, 1),
        dt.datetime(1969, 12, 31, 23, 59, 59),
        dt.datetime(1970, 1, 1),
        dt.datetime(2262, 4, 11),
        dt.datetime(2300, 1, 1),
    ]
    frames["temporal_range"] = pa.table(
        {
            "row": pa.array(range(len(extremes)), type=pa.int64()),
            "second": pa.array(extremes, type=pa.timestamp("s")),
            "date": pa.array([value.date() for value in extremes], type=pa.date32()),
        }
    )

    durations = [
        dt.timedelta(0),
        dt.timedelta(microseconds=1),
        dt.timedelta(microseconds=-1),
        dt.timedelta(days=1),
        dt.timedelta(days=-1),
        dt.timedelta(days=365, hours=6),
    ]
    frames["temporal_durations"] = pa.table(
        {
            "row": pa.array(range(len(durations)), type=pa.int64()),
            "value": pa.array(durations, type=pa.duration("us")),
        }
    )
    return frames


def key_frames() -> dict[str, pa.Table]:
    """Group by keys at three cardinalities, plus the awkward ones.

    The shape is db-benchmark's, at a size where the answer can be checked by hand
    rather than fingerprinted.
    """
    frames: dict[str, pa.Table] = {}
    for name, groups in (("keys_10", 10), ("keys_1000", 1000), ("keys_unique", TALL)):
        key = (
            np.arange(TALL, dtype=np.int64)
            if groups == TALL
            else below(splitmix64(SEED ^ salt(name), TALL), groups)
        )
        value = (splitmix64(SEED ^ salt(name + "value"), TALL) >> np.uint64(40)).astype(np.int64)
        frames[name] = pa.table(
            {
                "key": pa.array(key, type=pa.int64()),
                "value": pa.array(value, type=pa.int64()),
            }
        )

    # A null key and an empty string key, which group differently: pandas drops the
    # null by default and keeps the empty string, and a library that treats them
    # alike is wrong in one direction or the other.
    awkward = ["a", "b", None, "", "a", None, "", "b", "c", None]
    frames["keys_awkward"] = pa.table(
        {
            "key": pa.array(awkward, type=pa.large_string()),
            "value": pa.array(range(len(awkward)), type=pa.int64()),
        }
    )

    # Each column alone has duplicates and the pair is nearly unique, which is the
    # shape that tells a two column group by from two one column group bys.
    left = ["x", "x", "y", "y", "x", "y", "x", "y"]
    right = [1, 2, 1, 2, 1, 3, 3, 3]
    frames["keys_two_column"] = pa.table(
        {
            "left": pa.array(left, type=pa.large_string()),
            "right": pa.array(right, type=pa.int64()),
            "value": pa.array(range(len(left)), type=pa.int64()),
        }
    )
    return frames


def nested_frames() -> dict[str, pa.Table]:
    """List and struct columns, including the shapes pandas handles thinly."""
    lists = [[1, 2, 3], [], None, [None], [4], [5, None, 6], [7, 8, 9, 10]]
    structs = [
        {"a": 1, "b": "one"},
        {"a": None, "b": "two"},
        None,
        {"a": 3, "b": None},
        {"a": 4, "b": "four"},
        {"a": 5, "b": "five"},
        {"a": 6, "b": "six"},
    ]
    deep = [{"inner": {"deep": i}, "tag": f"t{i}"} for i in range(len(lists))]
    return {
        "nested_list": pa.table(
            {
                "row": pa.array(range(len(lists)), type=pa.int64()),
                "value": pa.array(lists, type=pa.large_list(pa.int64())),
            }
        ),
        "nested_struct": pa.table(
            {
                "row": pa.array(range(len(structs)), type=pa.int64()),
                "value": pa.array(
                    structs, type=pa.struct([("a", pa.int64()), ("b", pa.large_string())])
                ),
            }
        ),
        "nested_deep": pa.table(
            {
                "row": pa.array(range(len(deep)), type=pa.int64()),
                "value": pa.array(
                    deep,
                    type=pa.struct(
                        [
                            ("inner", pa.struct([("deep", pa.int64())])),
                            ("tag", pa.large_string()),
                        ]
                    ),
                ),
            }
        ),
    }


SECTIONS: dict[str, Callable[[], dict[str, pa.Table]]] = {
    "shape": shape_frames,
    "numeric": numeric_frames,
    "string": string_frames,
    "categorical": categorical_frames,
    "temporal": temporal_frames,
    "keys": key_frames,
    "nested": nested_frames,
}


def frames() -> dict[str, pa.Table]:
    """Builds every frame in the corpus.

    Returns:
        Frame name to Arrow table.
    """
    out: dict[str, pa.Table] = {}
    for build in SECTIONS.values():
        for name, table in build().items():
            if name in out:
                raise ValueError(f"two frames are called {name}")
            out[name] = table
    return out


def digest(table: pa.Table) -> str:
    """Returns a digest of a table's schema and values.

    Not a hash of the Arrow buffers and not a hash of the Parquet file. Both of
    those carry padding, compression settings and metadata that move between
    library versions, so either would report the corpus as changed on a pyarrow
    upgrade that changed nothing about the data. This hashes the schema and then
    the values as Python renders them, which keeps apart the things that have to be
    kept apart: a null is not a NaN, and a negative zero does not render as a zero.

    Args:
        table: The table.

    Returns:
        A hexadecimal SHA-256.
    """
    sha = hashlib.sha256()
    for field in table.schema:
        sha.update(f"{field.name}:{field.type}\n".encode())
    for column in table.columns:
        for value in column.to_pylist():
            sha.update(f"{value!r}\n".encode())
    return sha.hexdigest()


def describe(name: str, table: pa.Table) -> dict[str, Any]:
    """Describes one frame for the manifest.

    Args:
        name: The frame name.
        table: The table.

    Returns:
        The manifest record.
    """
    columns = [
        {"name": field.name, "type": str(field.type), "nulls": column.null_count}
        for field, column in zip(table.schema, table.columns, strict=True)
    ]
    record: dict[str, Any] = {
        "rows": table.num_rows,
        "columns": columns,
        "digest": digest(table),
    }
    if name.startswith(("float32", "float64")):
        # Written into the manifest rather than left in the source, so a report that
        # says a case failed at offset 3 can say it failed on the negative zero.
        record["edges"] = {key: offset for offset, key in enumerate(FLOAT_EDGES)}
    return record


def manifest() -> dict[str, Any]:
    """Builds the whole manifest.

    Returns:
        The document written to `corpus/manifest.json`.
    """
    built = frames()
    return {
        "seed": f"0x{SEED:016X}",
        "generator": "splitmix64 counter stream, matching tools/data.py in firepanda-bench",
        "pyarrow": ".".join(pa.__version__.split(".")[:2]),
        "frames": {name: describe(name, table) for name, table in sorted(built.items())},
    }


def dumps(doc: dict[str, Any]) -> str:
    """Renders the manifest as the committed file.

    Args:
        doc: The manifest.

    Returns:
        JSON with sorted keys and a trailing newline, so the diff of a corpus
        change is the change and not a reordering.
    """
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def write_frames(built: dict[str, pa.Table]) -> None:
    """Writes every frame to Arrow IPC under `corpus/`.

    Args:
        built: Frame name to table.
    """
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, table in built.items():
        with (
            pa.OSFile(str(CORPUS / f"{name}.arrow"), "wb") as sink,
            pa.ipc.new_file(sink, table.schema) as writer,
        ):
            writer.write_table(table)


def load(name: str) -> pa.Table:
    """Reads one frame back, generating the corpus first if it is not on disk.

    Args:
        name: The frame name.

    Returns:
        The table, with the schema it was written with.
    """
    path = CORPUS / f"{name}.arrow"
    if not path.exists():
        write_frames(frames())
    with pa.memory_map(str(path), "rb") as source:
        return pa.ipc.open_file(source).read_all()


def compare(doc: dict[str, Any], committed: dict[str, Any]) -> list[str]:
    """Returns the differences between a fresh manifest and the committed one.

    Args:
        doc: The manifest just generated.
        committed: The manifest read from disk.

    Returns:
        One line per difference, empty when they agree.
    """
    problems = []
    fresh, old = doc["frames"], committed.get("frames", {})
    for name in sorted(set(old) - set(fresh)):
        problems.append(f"{name}: in the manifest and not in the corpus")
    for name in sorted(set(fresh) - set(old)):
        problems.append(f"{name}: in the corpus and not in the manifest")
    for name in sorted(set(fresh) & set(old)):
        if fresh[name]["digest"] != old[name]["digest"]:
            problems.append(
                f"{name}: manifest says {old[name]['digest'][:12]}, "
                f"generated {fresh[name]['digest'][:12]}"
            )
        elif fresh[name] != old[name]:
            problems.append(f"{name}: same digest, different description")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Regenerates the corpus, or checks it against the committed manifest.

    Args:
        argv: Command line arguments.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify against the committed manifest"
    )
    parser.add_argument("--list", action="store_true", help="print the frame names")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(frames()):
            print(name)
        return 0

    built = frames()
    doc = manifest()
    write_frames(built)
    rows = sum(table.num_rows for table in built.values())
    print(f"{len(built)} frames, {rows} rows")

    if MANIFEST.exists():
        problems = compare(doc, json.loads(MANIFEST.read_text()))
        if problems:
            if args.check:
                print("the corpus does not match corpus/manifest.json:", file=sys.stderr)
                for line in problems:
                    print(f"  {line}", file=sys.stderr)
                print(
                    "run `pixi run corpus` and commit the manifest diff, which is the "
                    "record of what changed about the inputs",
                    file=sys.stderr,
                )
                return 1
        elif args.check:
            print("every digest matches corpus/manifest.json")
            return 0
    elif args.check:
        print("no corpus/manifest.json: run `pixi run corpus` and commit it", file=sys.stderr)
        return 1

    MANIFEST.write_text(dumps(doc))
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
