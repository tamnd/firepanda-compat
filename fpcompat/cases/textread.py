"""Reading text files with `read_csv`, `read_table` and `read_fwf`.

firepanda #1226 made `read_csv` take every one of pandas' arguments and added
`read_table` and `read_fwf`. A plain file with pandas' defaults still goes to the core
reader, and everything else goes to a Python reader that splits the text as pandas' C
parser does and types each column as it does.

Most cases read a fixed piece of text, because the point is how the text is read, and
the frame is only there because every case runs on one. The round trips write the frame
with `to_csv` and read it back, which is the most common use of the pair.

Every case runs in process, for the reason the pickle cases do.
"""

from __future__ import annotations

import io

from fpcompat.cases import case, section

section("textread")

IN_PROCESS_NOTE = (
    "the text readers live in firepanda's Python layer, which reads the text itself and "
    "hands the columns to the core as Arrow data, so a driver entry could only emit the "
    "frame it was handed and would be scoring itself"
)


def _csv(pd, text, **options):
    """Reads a piece of text with `read_csv`."""
    return pd.read_csv(io.StringIO(text), **options)


def _roundtrip(pd, frame, **options):
    """Writes a frame to a temporary file with `to_csv` and reads it back."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "frame.csv"
        frame.to_csv(target, index=False)
        return pd.read_csv(target, **options)


TEXT = "a,b,c\n1,x,2.5\n2,y,\n3,z,4.5\n"

READS = {
    "semicolons": ("a;b\n1;2\n3;4\n", {"sep": ";"}, ("sep",)),
    "whitespace": ("a  b\n1   2\n3 4\n", {"sep": r"\s+"}, ("sep",)),
    "no-header": ("1,2\n3,4\n", {"header": None}, ("header",)),
    "names": (TEXT, {"names": ["p", "q", "r"], "header": 0}, ("names", "header")),
    "skiprows": ("junk\n" + TEXT, {"skiprows": 1}, ("skiprows",)),
    "nrows": (TEXT, {"nrows": 2}, ("nrows",)),
    "dtype": (TEXT, {"dtype": {"a": "float64"}}, ("dtype",)),
    "text-dtype": ("a,b\n007,1\n", {"dtype": {"a": str}}, ("dtype",)),
    "converters": (TEXT, {"converters": {"a": lambda text: int(text) * 10}}, ("converters",)),
    "na-values": (TEXT, {"na_values": ["y"]}, ("na_values",)),
    "keep-default-na": ("a,b\nNA,1\n", {"keep_default_na": False}, ("keep_default_na",)),
    "true-values": (
        "a\nyes\nno\n",
        {"true_values": ["yes"], "false_values": ["no"]},
        ("true_values", "false_values"),
    ),
    "parse-dates": ("a,b\n2024-01-02,1\n2024-03-04,2\n", {"parse_dates": ["a"]}, ("parse_dates",)),
    "thousands": ('a\n"1,000"\n"2,500"\n', {"thousands": ","}, ("thousands",)),
    "decimal": ("a;b\n1,5;2\n", {"sep": ";", "decimal": ","}, ("decimal",)),
    "comment": ("a,b\n1,2 # note\n#skip\n3,4\n", {"comment": "#"}, ("comment",)),
    "quotechar": ("a,b\n'x,y',1\n", {"quotechar": "'"}, ("quotechar",)),
    "usecols-callable": (TEXT, {"usecols": lambda name: name != "b"}, ("usecols",)),
    "index-col": (TEXT, {"index_col": "b"}, ("index_col",)),
    "blank-names": ("a,,a\n1,2,3\n", {}, ()),
    "leading-labels": ("a,b\nx,1,2\ny,3,4\n", {}, ()),
    "short-rows": ("a,b,c\n1,2\n3,4,5\n", {}, ()),
    "padded-numbers": ("a,b\n 1, 2\n3,4\n", {}, ()),
    "bad-lines-skip": ("a,b\n1,2\n3,4,5\n6,7\n", {"on_bad_lines": "skip"}, ("on_bad_lines",)),
}

for name, (text, options, covers) in READS.items():
    case(
        f"textread/csv-{name}",
        "pandas.read_csv",
        frames=("two",),
        covers=covers,
        in_process=True,
        note=IN_PROCESS_NOTE,
        expr=lambda pd, df, text=text, options=options: _csv(pd, text, **options),
    )

case(
    "textread/csv-roundtrip",
    "pandas.read_csv",
    frames=("two", "tall", "integer_edges"),
    in_process=True,
    note="a frame written with to_csv and read back. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df),
)
case(
    "textread/csv-chunks",
    "pandas.read_csv",
    frames=("two",),
    covers=("chunksize",),
    in_process=True,
    note="each chunk is typed from its own rows and the labels carry on. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.concat(list(_csv(pd, "a\n1\n2\n3\n", chunksize=2))),
)
case(
    "textread/csv-bad-line",
    "pandas.read_csv",
    level="L4",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _csv(pd, "a,b\n1,2\n3,4,5\n"),
    raises=("ParserError", "Expected 2 fields in line 3, saw 3"),
)
case(
    "textread/csv-empty",
    "pandas.read_csv",
    level="L4",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _csv(pd, ""),
    raises=("EmptyDataError", "No columns to parse from file"),
)
case(
    "textread/table",
    "pandas.read_table",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_table(io.StringIO("a\tb\n1\tx\n2\ty\n")),
)
case(
    "textread/fwf-infer",
    "pandas.read_fwf",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(io.StringIO("  a    b  c\n  1  2.5  x\n 10  3.0  y\n")),
)
case(
    "textread/fwf-widths",
    "pandas.read_fwf",
    frames=("two",),
    covers=("widths",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(
        io.StringIO("  a    b  c\n  1  2.5  x\n 10  3.0  y\n"), widths=[3, 5, 3]
    ),
)
