"""Writing a frame or a column to a pickle and reading it back.

These cases were the pickle block in `fpcompat/cases/divergences.py` while firepanda
refused to pickle anything, on the grounds that reading a pickle runs whatever code is
in it. firepanda #1222 made frames, columns and indexes pickle through the Arrow data
they already export, with `to_pickle` and `read_pickle` taking pandas' parameters, so
the three cases moved here with new ids and the entry was retired. `read_pickle` warns
in its docstring to read only a trusted file, which is the warning pandas gives.

Each case is a real round trip through a temporary file rather than a name check,
because what the entry gave up was the round trip. The file itself is not compared: a
firepanda pickle names firepanda's classes and a pandas pickle names pandas', so
neither can read the other's, and what is compared is what each hands back.

Every case runs in process. The first run of this file sent them to the driver, which
has no entry for any of them, so the board after #1222 counted all thirteen runs as
unimplemented while each of them agreed with pandas.
"""

from __future__ import annotations

import pickle

from fpcompat.cases import case, section

section("pickle")

IN_PROCESS_NOTE = (
    "pickling lives in firepanda's Python layer, which copies the Arrow data the core "
    "exports into bytes and rebuilds it, so a driver entry could only emit the frame it "
    "was handed and would be scoring itself"
)


def _roundtrip(pd, value, name="frame.pkl", write=None, **options):
    """Writes a pickle to a temporary file and reads it back, with the same compression."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / name
        if write is None:
            value.to_pickle(target, **options)
        else:
            write(value, target, **options)
        return pd.read_pickle(target, compression=options.get("compression", "infer"))


case(
    "pickle/frame-roundtrip",
    "DataFrame.to_pickle",
    frames=("two", "tall"),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df),
)
case(
    "pickle/series-roundtrip",
    "Series.to_pickle",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df["b"]),
)
case(
    "pickle/read",
    "pandas.read_pickle",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df).shape,
)
case(
    "pickle/module-to-pickle",
    "pandas.to_pickle",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, write=pd.to_pickle),
)
case(
    "pickle/compression-by-ending",
    "DataFrame.to_pickle",
    frames=("two", "tall"),
    in_process=True,
    expr=lambda pd, df: _roundtrip(pd, df, name="frame.pkl.gz"),
    note="the compression is read from the file's ending, as it is for every writer. "
    + IN_PROCESS_NOTE,
)
case(
    "pickle/compression-by-name",
    "DataFrame.to_pickle",
    frames=("two",),
    covers=("compression",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, name="frame.bin", compression="bz2"),
)
case(
    "pickle/zip-archive",
    "Series.to_pickle",
    frames=("two",),
    covers=("compression",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df["a"], name="frame.pkl.zip"),
)
case(
    "pickle/protocol",
    "DataFrame.to_pickle",
    frames=("two",),
    covers=("protocol",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: _roundtrip(pd, df, protocol=2),
)
case(
    "pickle/dumps",
    "DataFrame.to_pickle",
    frames=("two", "tall"),
    note="the standard library's pickle rather than the method, which is how most code "
    "caches a frame. " + IN_PROCESS_NOTE,
    in_process=True,
    expr=lambda pd, df: pickle.loads(pickle.dumps(df)),
)
case(
    "pickle/dumps-column",
    "Series.to_pickle",
    frames=("two",),
    in_process=True,
    note=IN_PROCESS_NOTE,
    expr=lambda pd, df: pickle.loads(pickle.dumps(df["b"])),
)
