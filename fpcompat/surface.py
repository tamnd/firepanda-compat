"""The inventory of the pandas surface.

Implements `docs/specs/02-the-surface.md`.

Walks the twenty one pandas namespaces, keeps the public names, records every
callable with the parameters `inspect.signature` reports, and writes the result to
`surface/pandas-<version>.json`. The committed file is what the scoreboard counts
against, so the denominator in every published number comes from the pandas that
was installed rather than from anybody's memory of the documentation.

Three things about this tool are deliberate.

It builds a real object for every accessor namespace. `Series.str` is a descriptor
and reading it off the class gives an accessor factory rather than the accessor, so
the only way to see the fifty seven string methods is to make a string series and
ask it. A namespace whose sample object cannot be built is recorded as unavailable
with the exception text, and it is never silently dropped, because a namespace that
vanishes from the denominator is a namespace nobody has to conform to.

It records a parameter count that is sometimes a lower bound and says so. Some
pandas callables are C level and `inspect.signature` raises on them; those carry
`"signature": null` and are counted in `callables` and not in `parameters`.

It has a `--check` mode that regenerates and compares. The committed inventory
going stale is not a cosmetic problem: every coverage number is computed against
it, so a pandas upgrade that adds twelve methods has to show up as a diff in this
repository before it shows up as a coverage number that is quietly too high.

The inventory is a function of the interpreter and of pyarrow as well as of
pandas, which CI found rather than this comment predicting it. pandas 3.0.3 walked
on Python 3.12 has 3266 parameters and on 3.14 it has 3267, because
`Timestamp.fromisoformat` is a C level callable whose `object` parameter only
became visible to `inspect.signature` in 3.14. One parameter in three thousand is
exactly the size of drift that would otherwise be waved through, so the document
records all three versions, `--check` compares them before it compares the body
and says which one moved, and CI pins every one of them to what the committed file
names.

Usage:
    python -m fpcompat.surface                 # rewrite the committed inventory
    python -m fpcompat.surface --check         # fail if the committed file is stale
    python -m fpcompat.surface --gaps FILE.md  # names not mentioned in a markdown file
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SURFACE_DIR = ROOT / "surface"


def namespaces() -> dict[str, Any]:
    """Builds one sample object per pandas namespace.

    Accessors are descriptors, so `pandas.Series.str` is not the string namespace
    and only an actual string series can be asked what is in it. Each entry here is
    therefore a live object of the right dtype, built as cheaply as possible.

    Returns:
        A mapping from namespace name to the object to walk. A namespace whose
        sample object cannot be built is present with the exception as its value,
        so that the caller records it as unavailable rather than losing it.
    """
    import pandas as pd
    import pyarrow as pa

    def build(fn):
        try:
            return fn()
        except Exception as exc:  # recorded, not swallowed
            return exc

    return {
        "pandas": build(lambda: pd),
        "DataFrame": build(lambda: pd.DataFrame),
        "Series": build(lambda: pd.Series),
        "Index": build(lambda: pd.Index),
        "MultiIndex": build(lambda: pd.MultiIndex),
        "DatetimeIndex": build(lambda: pd.DatetimeIndex),
        "str": build(lambda: pd.Series(["a"], dtype="str").str),
        "dt": build(lambda: pd.Series(pd.to_datetime(["2026-01-01"])).dt),
        "cat": build(lambda: pd.Series(["a"], dtype="category").cat),
        "list": build(lambda: pd.Series([[1]], dtype=pd.ArrowDtype(pa.list_(pa.int64()))).list),
        "struct": build(
            lambda: (
                pd.Series([{"a": 1}], dtype=pd.ArrowDtype(pa.struct([("a", pa.int64())]))).struct
            )
        ),
        "GroupBy": build(lambda: pd.DataFrame({"a": [1], "b": [1]}).groupby("a")),
        "Rolling": build(lambda: pd.DataFrame({"a": [1.0]}).rolling(1)),
        "Expanding": build(lambda: pd.DataFrame({"a": [1.0]}).expanding()),
        "ExponentialMovingWindow": build(lambda: pd.DataFrame({"a": [1.0]}).ewm(com=1)),
        "Resampler": build(
            lambda: pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-01-01"])).resample("D")
        ),
        "Timestamp": build(lambda: pd.Timestamp("2026-01-01")),
        "Timedelta": build(lambda: pd.Timedelta("1D")),
        "offsets": build(lambda: pd.offsets),
        "errors": build(lambda: __import__("pandas.errors", fromlist=["x"])),
        "api.types": build(lambda: __import__("pandas.api.types", fromlist=["x"])),
    }


ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def render_default(value: Any) -> str:
    """Renders a parameter default so that two runs produce the same bytes.

    pandas uses bare sentinel objects for several defaults, `lib.no_default` among
    them, and their repr carries the address they happen to have been allocated at.
    Left alone that makes the inventory differ between two runs of the same tool
    against the same pandas, which turns `--check` into noise and a diff in review
    into nothing.

    Args:
        value: The default.

    Returns:
        The repr with any address removed.
    """
    return ADDRESS.sub("", repr(value))


def signature_of(obj: Any) -> list[dict[str, Any]] | None:
    """Reads the parameters of a callable, or reports that it has none to read.

    Args:
        obj: The callable.

    Returns:
        One entry per parameter, with its name, kind and default rendered as a
        string, or None when `inspect.signature` refuses. `self` is dropped, since
        a user never passes it and counting it would inflate every method by one.
    """
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    out = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        out.append(
            {
                "name": name,
                "kind": param.kind.name,
                "default": None
                if param.default is inspect.Parameter.empty
                else render_default(param.default),
            }
        )
    return out


def walk(name: str, obj: Any) -> dict[str, Any]:
    """Records every public member of one namespace.

    Args:
        name: The namespace name, used only in the record.
        obj: The sample object, or the exception raised while building it.

    Returns:
        The namespace record: the public names, the callables with their
        parameters, the properties, and the three counts the scoreboard uses.
    """
    if isinstance(obj, Exception):
        return {
            "namespace": name,
            "available": False,
            "reason": f"{type(obj).__name__}: {obj}",
            "names": [],
            "members": {},
            "counts": {"names": 0, "callables": 0, "parameters": 0, "properties": 0},
        }

    owner = obj if isinstance(obj, type) else type(obj)
    names = sorted(n for n in dir(obj) if not n.startswith("_"))
    members: dict[str, Any] = {}
    callables = parameters = properties = 0
    unreadable = []

    for member in names:
        try:
            value = getattr(obj, member)
        except Exception as exc:  # a name that raises on access is still a name
            members[member] = {"kind": "unreadable", "reason": type(exc).__name__}
            continue
        on_owner = getattr(owner, member, None)
        if isinstance(on_owner, property):
            members[member] = {"kind": "property"}
            properties += 1
        elif callable(value):
            params = signature_of(value)
            members[member] = {"kind": "callable", "signature": params}
            callables += 1
            if params is None:
                unreadable.append(member)
            else:
                parameters += len(params)
        else:
            members[member] = {"kind": "attribute", "type": type(value).__name__}

    record = {
        "namespace": name,
        "available": True,
        "names": names,
        "members": members,
        "counts": {
            "names": len(names),
            "callables": callables,
            "parameters": parameters,
            "properties": properties,
        },
    }
    if unreadable:
        # The parameter count is a lower bound whenever this list is non empty, and
        # the file says which names made it one rather than leaving the reader to
        # wonder why the arithmetic does not work out.
        record["parameters_are_a_lower_bound_because"] = unreadable
    return record


def inventory() -> dict[str, Any]:
    """Builds the whole inventory.

    Returns:
        The document that gets written to `surface/pandas-<version>.json`.
    """
    import pandas as pd
    import pyarrow as pa

    spaces = {name: walk(name, obj) for name, obj in namespaces().items()}
    totals = {"names": 0, "callables": 0, "parameters": 0, "properties": 0}
    for record in spaces.values():
        for key in totals:
            totals[key] += record["counts"][key]

    # pandas is recorded exactly because it is the subject. The other two are
    # recorded to the minor version, because they are pins for the environment
    # rather than the thing being measured, and a patch release of either that
    # genuinely moved a signature would still fail the comparison of the body.
    # Recording their patch versions instead would mean a diff every time a
    # contributor's lockfile picked up pyarrow 25.0.1 over 25.0.0.
    return {
        "pandas": pd.__version__,
        "pyarrow": ".".join(pa.__version__.split(".")[:2]),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "totals": totals,
        "namespaces": spaces,
    }


def path_for(version: str) -> Path:
    """Returns the committed inventory path for a pandas version.

    Args:
        version: The pandas version string.

    Returns:
        The path, which is one file per pandas version so that an upgrade is a new
        file beside the old one rather than a rewrite of history.
    """
    return SURFACE_DIR / f"pandas-{version}.json"


def dumps(doc: dict[str, Any]) -> str:
    """Renders the inventory the one way it is ever rendered.

    Args:
        doc: The inventory.

    Returns:
        JSON with sorted keys and a trailing newline, so that `--check` compares
        bytes and a diff in review is a diff in the surface rather than in the
        formatting.
    """
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def names_in(doc: dict[str, Any]) -> set[str]:
    """Returns every public name in an inventory, across all namespaces.

    Args:
        doc: The inventory.

    Returns:
        The set of names. Namespaces are not kept apart here, because the caller
        that wants this is asking whether a name is mentioned anywhere.
    """
    out: set[str] = set()
    for record in doc["namespaces"].values():
        out.update(record["names"])
    return out


def gaps(doc: dict[str, Any], markdown: Path) -> dict[str, list[str]]:
    """Finds pandas names a markdown document never mentions.

    This is how `docs/specs/02-the-surface.md` was written and how it stays true.
    The parity checklist in the library repository names symbols in backticks, so
    the parser takes every backticked token, adds the last dotted component of each,
    and subtracts that from the inventory.

    It is a blunt instrument on purpose: a name mentioned in passing counts as
    mentioned, which biases the answer towards saying the document covers more than
    it does. A gap it reports is therefore a real gap.

    Args:
        doc: The inventory.
        markdown: The document to read.

    Returns:
        A mapping from namespace to the sorted names that document does not
        mention, with empty namespaces dropped.
    """
    text = markdown.read_text()
    mentioned = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text))
    mentioned |= {name.split(".")[-1] for name in mentioned}
    out = {}
    for space, record in doc["namespaces"].items():
        missing = sorted(n for n in record["names"] if n not in mentioned)
        if missing:
            out[space] = missing
    return out


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Args:
        argv: Arguments, defaulting to `sys.argv`.

    Returns:
        A process exit code. `--check` returns 1 when the committed inventory does
        not match the installed pandas, which is a failure with a fix attached: run
        the tool without the flag and commit the diff.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed inventory is missing or stale",
    )
    parser.add_argument(
        "--gaps",
        type=Path,
        help="list pandas names a markdown document does not mention",
    )
    args = parser.parse_args(argv)

    doc = inventory()

    if args.gaps:
        missing = gaps(doc, args.gaps)
        total = sum(len(v) for v in missing.values())
        print(f"{args.gaps} mentions {len(names_in(doc)) - total} of {len(names_in(doc))} names\n")
        for space, names in missing.items():
            print(f"{space}: {len(names)} not mentioned")
            print("  " + ", ".join(names))
        return 0

    target = path_for(doc["pandas"])

    if args.check:
        if not target.exists():
            print(
                f"no committed inventory for pandas {doc['pandas']}: "
                f"run `pixi run surface` and commit {target.relative_to(ROOT)}",
                file=sys.stderr,
            )
            return 1
        committed = json.loads(target.read_text())
        if committed.get("python") != doc["python"]:
            print(
                f"{target.relative_to(ROOT)} was generated on Python "
                f"{committed.get('python')} and this is Python {doc['python']}. The "
                "inventory is a function of the interpreter as well as of pandas, so "
                "regenerate it on the pinned interpreter or pin this environment to "
                "the one it names.",
                file=sys.stderr,
            )
            return 1
        if committed.get("pyarrow") != doc["pyarrow"]:
            print(
                f"{target.relative_to(ROOT)} was generated against pyarrow "
                f"{committed.get('pyarrow')} and this is {doc['pyarrow']}. The nested "
                "accessors are built on Arrow dtypes, so this moves the surface.",
                file=sys.stderr,
            )
            return 1
        if target.read_text() != dumps(doc):
            print(
                f"{target.relative_to(ROOT)} does not match the installed pandas: "
                "run `pixi run surface` and commit the diff",
                file=sys.stderr,
            )
            return 1
        counts = doc["totals"]
        print(
            f"pandas {doc['pandas']}: {counts['names']} names, "
            f"{counts['callables']} callables, {counts['parameters']} parameters, current"
        )
        return 0

    SURFACE_DIR.mkdir(exist_ok=True)
    target.write_text(dumps(doc))
    counts = doc["totals"]
    print(
        f"pandas {doc['pandas']}: {counts['names']} names, {counts['callables']} callables, "
        f"{counts['parameters']} parameters, {counts['properties']} properties"
    )
    print(f"wrote {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
