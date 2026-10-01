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

import csv
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
    "delimiter": ("a|b\n1|2\n", {"delimiter": "|"}, ("delimiter",)),
    "python-engine": (TEXT, {"engine": "python"}, ("engine",)),
    "c-engine-options": (
        TEXT,
        {"engine": "c", "low_memory": False, "memory_map": False},
        ("engine", "low_memory", "memory_map"),
    ),
    "skipinitialspace": ("a, b\n1, 2\n", {"skipinitialspace": True}, ("skipinitialspace",)),
    "skipfooter": (
        "a,b\n1,2\n3,4\nfooter\n",
        {"skipfooter": 1, "engine": "python"},
        ("skipfooter", "engine"),
    ),
    "na-filter-off": ("a,b\nNA,1\n", {"na_filter": False}, ("na_filter",)),
    "blank-lines-kept": ("a,b\n1,2\n\n3,4\n", {"skip_blank_lines": False}, ("skip_blank_lines",)),
    "date-format": (
        "a\n02/01/2024\n03/04/2024\n",
        {"parse_dates": ["a"], "date_format": "%d/%m/%Y"},
        ("parse_dates", "date_format"),
    ),
    "dayfirst": (
        "a\n02/01/2024\n13/04/2024\n",
        {"parse_dates": ["a"], "dayfirst": True},
        ("parse_dates", "dayfirst"),
    ),
    "cache-dates-off": (
        "a\n2024-01-02\n",
        {"parse_dates": ["a"], "cache_dates": False},
        ("parse_dates", "cache_dates"),
    ),
    "lineterminator": ("a,b~1,2~3,4~", {"lineterminator": "~"}, ("lineterminator",)),
    "quote-all": ('a,b\n"x",1\n', {"quoting": csv.QUOTE_ALL}, ("quoting",)),
    "doublequote": ('a,b\n"x""y",1\n', {"doublequote": True}, ("doublequote",)),
    "escapechar": (
        'a,b\n"x\\"y",1\n',
        {"doublequote": False, "escapechar": "\\"},
        ("doublequote", "escapechar"),
    ),
    "round-trip-precision": (
        "a,b\n1.123456789012345678,2\n",
        {"float_precision": "round_trip"},
        ("float_precision",),
    ),
    "nullable-backend": (TEXT, {"dtype_backend": "numpy_nullable"}, ("dtype_backend",)),
    "arrow-backend": (TEXT, {"dtype_backend": "pyarrow"}, ("dtype_backend",)),
    "encoding-errors": (
        "a,b\n1,2\n",
        {"encoding": "utf-8", "encoding_errors": "strict"},
        ("encoding", "encoding_errors"),
    ),
    "no-compression": ("a,b\n1,2\n", {"compression": None}, ("compression",)),
    "dialect": ("a;b\n1;2\n", {"dialect": csv.excel, "sep": ";"}, ("dialect", "sep")),
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

for name, (text, options, covers) in READS.items():
    case(
        f"textread/table-{name}",
        "pandas.read_table",
        frames=("two",),
        covers=covers,
        in_process=True,
        note="the read_csv table read with read_table, a comma given as the separator. "
        + IN_PROCESS_NOTE,
        expr=lambda pd, df, text=text, options=options: pd.read_table(
            io.StringIO(text), **({} if "delimiter" in options else {"sep": ","}) | options
        ),
    )

DATED = '{"a":{"0":1,"1":2},"b":{"0":"2024-01-02","1":"2024-01-03"},"c":{"0":1.5,"1":2.25}}'
FLAGGED = '[{"a":1,"b":"x","c":1.5,"d":true},{"a":null,"b":null,"c":null,"d":false}]'

JSONS = {
    "dtype": (DATED, {"dtype": {"a": "float64"}}, ("dtype",)),
    "dtype-off": (DATED, {"dtype": False}, ("dtype",)),
    "convert-axes-off": (DATED, {"convert_axes": False}, ("convert_axes",)),
    "named-dates": (
        DATED,
        {"convert_dates": ["b"], "keep_default_dates": False},
        ("convert_dates", "keep_default_dates"),
    ),
    "default-dates-off": (
        '{"modified":{"0":1704153600000}}',
        {"keep_default_dates": False},
        ("keep_default_dates",),
    ),
    "precise-float": (
        '{"a":{"0":1.123456789012345678}}',
        {"precise_float": True},
        ("precise_float",),
    ),
    "date-unit": (
        '{"a":{"0":1704153600}}',
        {"convert_dates": ["a"], "date_unit": "s"},
        ("convert_dates", "date_unit"),
    ),
    "encoding-errors": (
        DATED,
        {"encoding": "utf-8", "encoding_errors": "strict"},
        ("encoding", "encoding_errors"),
    ),
    "nullable-backend": (FLAGGED, {"dtype_backend": "numpy_nullable"}, ("dtype_backend",)),
    "arrow-backend": (FLAGGED, {"dtype_backend": "pyarrow"}, ("dtype_backend",)),
    "arrow-backend-dates": (
        DATED,
        {"dtype_backend": "pyarrow", "convert_dates": ["b"]},
        ("dtype_backend", "convert_dates"),
    ),
}

for name, (text, options, covers) in JSONS.items():
    case(
        f"textread/json-{name}",
        "pandas.read_json",
        frames=("two",),
        covers=covers,
        in_process=True,
        note=IN_PROCESS_NOTE,
        expr=lambda pd, df, text=text, options=options: pd.read_json(io.StringIO(text), **options),
    )

case(
    "textread/json-series-backend",
    "pandas.read_json",
    frames=("two",),
    covers=("typ", "dtype_backend"),
    in_process=True,
    note="a column read with a gap, in Arrow's integer type. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_json(
        io.StringIO('{"0":1,"1":null}'), typ="series", dtype_backend="pyarrow"
    ),
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
    "textread/csv-iterator",
    "pandas.read_csv",
    frames=("two",),
    covers=("iterator",),
    in_process=True,
    note="a reader handed back and asked for two rows. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_csv(io.StringIO(TEXT), iterator=True).get_chunk(2),
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

LINES = '{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n{"a":3,"b":"z"}\n'
FIXED = "a  b\n1  x\n22 y\n"

case(
    "textread/json-lines-nrows",
    "pandas.read_json",
    frames=("two",),
    covers=("path_or_buf", "lines", "nrows"),
    in_process=True,
    note="the first two lines of a JSON lines text. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_json(io.StringIO(LINES), lines=True, nrows=2),
)
case(
    "textread/json-lines-chunks",
    "pandas.read_json",
    frames=("two",),
    covers=("lines", "chunksize"),
    in_process=True,
    note="each chunk read from its own lines, the labels carried on. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: [
        chunk for chunk in pd.read_json(io.StringIO(LINES), lines=True, chunksize=2)
    ][1],
)
case(
    "textread/json-lines-chunk-rows",
    "pandas.read_json",
    frames=("two",),
    covers=("lines", "chunksize", "nrows"),
    in_process=True,
    note="nrows is looked at before each chunk, so the last chunk is read whole. "
    + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.concat(
        list(pd.read_json(io.StringIO(LINES * 2), lines=True, chunksize=2, nrows=3))
    ),
)
case(
    "textread/json-lines-read-all",
    "pandas.read_json",
    frames=("two",),
    covers=("lines", "chunksize"),
    in_process=True,
    note="the chunk reader asked for everything at once. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_json(io.StringIO(LINES), lines=True, chunksize=1).read(),
)
case(
    "textread/json-plain-options",
    "pandas.read_json",
    frames=("two",),
    covers=("lines", "compression", "engine"),
    in_process=True,
    note="no compression and the default engine named. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_json(
        io.StringIO(LINES), lines=True, compression=None, engine="ujson"
    ),
)
case(
    "textread/json-chunk-size-bad",
    "pandas.read_json",
    level="L4",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_json(io.StringIO(LINES), lines=True, chunksize=0),
    raises=("ValueError", "must be an integer >=1"),
)
case(
    "textread/table-iterator",
    "pandas.read_table",
    frames=("two",),
    covers=("filepath_or_buffer", "iterator"),
    in_process=True,
    note="a reader handed back and asked for one row. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_table(io.StringIO("a\tb\n1\t2\n3\t4\n"), iterator=True).get_chunk(
        1
    ),
)
case(
    "textread/table-chunks",
    "pandas.read_table",
    frames=("two",),
    covers=("chunksize",),
    in_process=True,
    note="each chunk typed from its own rows, the labels carried on. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.concat(
        list(pd.read_table(io.StringIO("a\tb\n1\t2\n3\tx\n"), chunksize=1))
    ),
)
case(
    "textread/fwf-colspecs",
    "pandas.read_fwf",
    frames=("two",),
    covers=("filepath_or_buffer", "colspecs"),
    in_process=True,
    note="the fields cut where the spans say. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(io.StringIO(FIXED), colspecs=[(0, 2), (3, 4)]),
)
case(
    "textread/fwf-infer-rows",
    "pandas.read_fwf",
    frames=("two",),
    covers=("infer_nrows",),
    in_process=True,
    note="the fields found from the first row alone. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(io.StringIO(FIXED), infer_nrows=1),
)
case(
    "textread/fwf-iterator",
    "pandas.read_fwf",
    frames=("two",),
    covers=("iterator",),
    in_process=True,
    note="a reader handed back and asked for one row. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(io.StringIO(FIXED), iterator=True).get_chunk(1),
)
case(
    "textread/fwf-chunks",
    "pandas.read_fwf",
    frames=("two",),
    covers=("chunksize",),
    in_process=True,
    note="each chunk typed from its own rows, the labels carried on. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.concat(list(pd.read_fwf(io.StringIO(FIXED), chunksize=1))),
)
case(
    "textread/fwf-read-options",
    "pandas.read_fwf",
    frames=("two",),
    covers=("widths", "kwds"),
    in_process=True,
    note="read_csv's header and skiprows handed through. " + IN_PROCESS_NOTE,
    expr=lambda pd, df: pd.read_fwf(io.StringIO(FIXED), widths=[3, 1], header=None, skiprows=1),
)
