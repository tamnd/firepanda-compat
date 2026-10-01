"""Writing a frame to Parquet, Feather and ORC and reading it back.

firepanda #1224 added `to_parquet`, `to_feather` and `to_orc` with `read_parquet`,
`read_feather` and `read_orc`. They go through pyarrow, imported only when one of them is
called, and write the `pandas` metadata key in the layout pyarrow writes for pandas, so a
file keeps its row labels, types and `attrs` across the two libraries.

Each case is a real round trip through a temporary file, because a file that pandas
cannot read back the same way is what these functions exist to prevent. The file's bytes
are not compared, since the metadata names the library that wrote it.

Every case runs in process, for the reason the pickle cases do.
"""

from __future__ import annotations

from fpcompat.cases import case, section

section("columnar")

IN_PROCESS_NOTE = (
    "the file formats live in firepanda's Python layer, which hands the Arrow stream the "
    "core exports to pyarrow, so a driver entry could only emit the frame it was handed "
    "and would be scoring itself"
)


def _roundtrip(pd, frame, kind="parquet", name=None, read=None, **options):
    """Writes a frame to a temporary file and reads it back with the matching reader."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / (name or f"frame.{kind}")
        getattr(frame, f"to_{kind}")(target, **options)
        return getattr(pd, f"read_{kind}")(target, **(read or {}))


FRAMES = ("two", "tall", "integer_edges", "strings_unicode")

case(
    "columnar/parquet-roundtrip",
    "DataFrame.to_parquet",
    frames=FRAMES,
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df),
)
case(
    "columnar/parquet-read-columns",
    "pandas.read_parquet",
    frames=("two",),
    covers=("columns",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, read={"columns": ["c", "a"]}),
)
case(
    "columnar/parquet-labels",
    "DataFrame.to_parquet",
    frames=("two",),
    covers=("index",),
    in_process=True,
    expr=lambda pd, df: _roundtrip(pd, df.set_index("c")),
    note="row labels that are not a range are stored as a column and named in the "
    "metadata, and come back as the index. " + IN_PROCESS_NOTE,
)
case(
    "columnar/parquet-no-labels",
    "DataFrame.to_parquet",
    frames=("two",),
    covers=("index",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df.set_index("c"), index=False),
)
case(
    "columnar/parquet-compression",
    "DataFrame.to_parquet",
    frames=("tall",),
    covers=("compression",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, compression="zstd"),
)
case(
    "columnar/parquet-bytes",
    "DataFrame.to_parquet",
    frames=("two",),
    in_process=True,
    note="with no path the file comes back as bytes. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_parquet(__import__("io").BytesIO(df.to_parquet())),
)
case(
    "columnar/feather-roundtrip",
    "DataFrame.to_feather",
    frames=FRAMES,
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="feather"),
)
case(
    "columnar/feather-read-columns",
    "pandas.read_feather",
    frames=("two",),
    covers=("columns",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="feather", read={"columns": ["b"]}),
)
case(
    "columnar/orc-roundtrip",
    "DataFrame.to_orc",
    frames=("two", "tall"),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="orc"),
)
case(
    "columnar/orc-read-columns",
    "pandas.read_orc",
    frames=("two",),
    covers=("columns",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="orc", read={"columns": ["a", "c"]}),
)


def _gapped(pd):
    """A frame with a gap in a number, a text and a flag column, all with values."""
    return pd.DataFrame(
        {
            "i": pd.Series([1.0, float("nan"), 3.0]),
            "j": pd.Series([1, 2, 3]),
            "s": pd.Series(["x", float("nan"), "z"]),
            "b": pd.Series([True, False, True]),
        }
    )


def _partitioned(pd):
    """The folders `partition_cols` writes, and the frame read back from them."""
    import os
    import tempfile

    frame = pd.DataFrame({"k": ["x", "y", "x"], "v": [1, 2, 3]})
    with tempfile.TemporaryDirectory() as folder:
        frame.to_parquet(folder, partition_cols=["k"])
        back = pd.read_parquet(folder).sort_values("v").reset_index(drop=True)
        return sorted(os.listdir(folder)), back.astype({"k": str})


case(
    "columnar/parquet-nullable-backend",
    "pandas.read_parquet",
    frames=("single",),
    covers=("path", "dtype_backend"),
    in_process=True,
    note="numbers, text and flags read into pandas' nullable types. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, _gapped(pd), read={"dtype_backend": "numpy_nullable"}),
)
case(
    "columnar/parquet-arrow-backend",
    "pandas.read_parquet",
    frames=("single",),
    covers=("dtype_backend",),
    in_process=True,
    note="every column read as an ArrowDtype of its Arrow type. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, _gapped(pd), read={"dtype_backend": "pyarrow"}),
)
case(
    "columnar/parquet-read-options",
    "pandas.read_parquet",
    frames=("two",),
    covers=("engine", "filesystem", "to_pandas_kwargs", "kwargs"),
    in_process=True,
    note="the engine named, no file system, empty to_pandas_kwargs and a pyarrow "
    "option handed through. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(
        pd,
        df,
        read={
            "engine": "pyarrow",
            "filesystem": None,
            "to_pandas_kwargs": {},
            "use_threads": False,
        },
    ),
)
case(
    "columnar/parquet-filters",
    "pandas.read_parquet",
    frames=("single",),
    covers=("filters",),
    in_process=True,
    note="only the rows the filter keeps. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(
        pd, pd.DataFrame({"a": [1, 2, 3]}), read={"filters": [("a", ">", 1)]}
    ),
)
case(
    "columnar/parquet-write-options",
    "DataFrame.to_parquet",
    frames=("two",),
    covers=("path", "engine", "filesystem", "kwargs"),
    in_process=True,
    note="the engine named, no file system and a pyarrow option handed through. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, engine="pyarrow", filesystem=None, row_group_size=1),
)
case(
    "columnar/parquet-partitions",
    "DataFrame.to_parquet",
    frames=("single",),
    covers=("partition_cols",),
    in_process=True,
    note="one folder per key, read back as a frame. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _partitioned(pd),
)
case(
    "columnar/feather-backends",
    "pandas.read_feather",
    frames=("single",),
    covers=("path", "dtype_backend"),
    in_process=True,
    note="the nullable and the Arrow types of the same file. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: [
        _roundtrip(pd, _gapped(pd), kind="feather", read={"dtype_backend": backend})
        .dtypes.astype(str)
        .tolist()
        for backend in ("numpy_nullable", "pyarrow")
    ],
)
case(
    "columnar/feather-threads",
    "pandas.read_feather",
    frames=("two",),
    covers=("use_threads",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="feather", read={"use_threads": False}),
)
case(
    "columnar/feather-write-options",
    "DataFrame.to_feather",
    frames=("two",),
    covers=("path", "kwargs"),
    in_process=True,
    note="a pyarrow option handed through. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, kind="feather", compression="uncompressed"),
)
case(
    "columnar/orc-backends",
    "pandas.read_orc",
    frames=("single",),
    covers=("path", "dtype_backend"),
    in_process=True,
    note="the nullable and the Arrow types of the same file. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: [
        _roundtrip(pd, _gapped(pd), kind="orc", read={"dtype_backend": backend})
        .dtypes.astype(str)
        .tolist()
        for backend in ("numpy_nullable", "pyarrow")
    ],
)
case(
    "columnar/orc-write-options",
    "DataFrame.to_orc",
    frames=("two",),
    covers=("path", "engine", "index", "engine_kwargs"),
    in_process=True,
    note="the engine named, no labels and a pyarrow option handed through. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(
        pd, df, kind="orc", engine="pyarrow", index=False, engine_kwargs={"compression": "zlib"}
    ),
)
