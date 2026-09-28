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
