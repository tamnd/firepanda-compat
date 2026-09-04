"""Runs cases against an engine and writes a result file.

Implements the runner from `docs/specs/03-harness.md`.

Four outcomes and there is no fifth. `pass`, `fail`, `divergent`, `unimplemented`. A
case that has never run is not a pass, and every conformance suite that ever lied did
it by turning failures into skips, so the way to not do that is to have no skip
outcome at all. `unimplemented` is a real outcome with a real meaning, which is that
the name does not exist yet, and it counts in the denominator exactly as hard as a
failure does.

One engine per process, always, even when both could import. pandas leaves global
state behind, options are process wide, and a firepanda extension module and pandas
in one interpreter would share an allocator and a set of Arrow symbols. Isolation
also means a segfault in the subject is a failed case with a captured signal rather
than a lost run, which matters more than it should while the subject is a young
library in a systems language.

The protocol between the runner and its worker is one JSON object per line. The
worker announces a case before it runs it and reports the result after, so when the
worker dies the runner knows exactly which case killed it, records that case as a
failure with the signal in the reason, and starts a fresh worker on what is left. A
crash is a conformance failure of the loudest kind and it is never a skip.

Usage:
    python -m fpcompat.runner --engine pandas --oracle
    python -m fpcompat.runner --engine firepanda --filter strings
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fpcompat import corpus
from fpcompat.cases import NO_WARNING, Case, registry, select
from fpcompat.compare import check_error, check_warnings, compare
from fpcompat.engines import load

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PASS = "pass"
FAIL = "fail"
DIVERGENT = "divergent"
UNIMPLEMENTED = "unimplemented"
OUTCOMES = (PASS, FAIL, DIVERGENT, UNIMPLEMENTED)


def divergent_ids() -> frozenset[str]:
    """The case ids registered as divergences.

    The registry itself lands with issue #7. Until it does this is empty, which is the
    correct behaviour rather than a placeholder: with no registered divergences, a
    case that diverges is a failure, which is the strict reading and the one that
    cannot flatter anybody.

    Returns:
        The registered ids.
    """
    try:
        # Imported here rather than at the top because the module does not exist yet.
        from fpcompat.divergences import case_ids
    except ImportError:
        return frozenset()
    return case_ids()


def _unimplemented(error: BaseException) -> bool:
    """Whether an exception means the name does not exist yet.

    Exactly one condition per document 03: the subject raising `AttributeError` or
    `NotImplementedError` at the top of the call. The depth check is what separates
    those two from the same exception thrown from three frames inside a half written
    method, which is a fail on purpose, because a method that works for two of its
    five parameter values is worse than an absent one and the score should say so.

    The depth is measured in traceback frames below the case expression. An attribute
    that does not resolve raises inside the lambda itself and has one frame. A method
    body whose whole content is `raise NotImplementedError` has two. Anything deeper
    got somewhere before it gave up.

    Args:
        error: What was raised.

    Returns:
        Whether this counts as unimplemented rather than as a failure.
    """
    depth = len(traceback.extract_tb(error.__traceback__))
    if isinstance(error, AttributeError):
        return depth <= 2
    if isinstance(error, NotImplementedError):
        return depth <= 3
    return False


def _run_expression(
    case: Case, engine: Any, frame_name: str
) -> tuple[Any, BaseException | None, list[Any]]:
    """Evaluates one case expression, capturing what it raised and what it warned.

    Args:
        case: The case.
        engine: The engine.
        frame_name: The corpus frame.

    Returns:
        The answer, the exception if there was one, and the warnings caught.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            frame = engine.frame(frame_name)
            return case.expr(engine.module(), frame), None, list(caught)
        except BaseException as error:  # noqa: BLE001  a crash in the subject is a result
            return None, error, list(caught)


def run_case(case: Case, oracle: Any, subject: Any, frame_name: str) -> dict[str, Any]:
    """Runs one case on one frame against both engines.

    Args:
        case: The case.
        oracle: The pandas engine.
        subject: The engine under test, which is pandas again in oracle mode.
        frame_name: The corpus frame.

    Returns:
        The result record.
    """
    started = time.perf_counter()
    record: dict[str, Any] = {
        "id": case.id,
        "frame": frame_name,
        "api": case.api,
        "section": case.section,
        "level": case.level,
        "outcome": PASS,
        "detail": "",
        "relaxations_used": [],
    }

    expected, expected_error, expected_warnings = _run_expression(case, oracle, frame_name)
    actual, actual_error, actual_warnings = _run_expression(case, subject, frame_name)
    record["seconds"] = round(time.perf_counter() - started, 6)

    if case.id in divergent_ids():
        record["outcome"] = DIVERGENT
        record["detail"] = "registered as a divergence"
        return record

    if actual_error is not None and _unimplemented(actual_error):
        record["outcome"] = UNIMPLEMENTED
        record["detail"] = f"{type(actual_error).__name__}: {actual_error}"
        return record

    if case.raises is not None:
        # An L4 case. Both engines have to fail the same way, and the oracle failing
        # differently from what the case declared is a broken case rather than a
        # conformance result, so it is reported as such.
        oracle_verdict = check_error(expected_error, *case.raises)
        if not oracle_verdict:
            record["outcome"] = FAIL
            record["detail"] = f"the case is wrong about pandas: {oracle_verdict.summary()}"
            return record
        verdict = check_error(actual_error, *case.raises)
        if not verdict:
            record["outcome"] = FAIL
            record["detail"] = verdict.summary()
            return record
        return _check_warnings(case, expected_warnings, actual_warnings, record)

    if expected_error is not None:
        record["outcome"] = FAIL
        record["detail"] = (
            f"pandas raised {type(expected_error).__name__}: {expected_error}. The case "
            "does not declare an exception, so either the expression is wrong or the "
            "case should be an L4 one"
        )
        return record

    if actual_error is not None:
        record["outcome"] = FAIL
        record["detail"] = f"raised {type(actual_error).__name__}: {actual_error}"
        return record

    try:
        verdict = compare(expected, actual, case.rules)
    except Exception as error:  # noqa: BLE001  a broken comparison is reported, not raised
        # The comparison raising is a bug in this repository rather than a conformance
        # result, and it is reported as a failure with the traceback so that it is
        # loud. What it must not do is take the worker down and lose the six hundred
        # results either side of it.
        record["outcome"] = FAIL
        record["detail"] = (
            f"the comparison itself raised {type(error).__name__}: {error}. That is a "
            "bug in fpcompat.compare and not a fact about either engine"
        )
        return record
    record["relaxations_used"] = sorted(verdict.relaxations_used)
    if not verdict:
        record["outcome"] = FAIL
        record["detail"] = verdict.summary()
        return record
    return _check_warnings(case, expected_warnings, actual_warnings, record)


def _check_warnings(
    case: Case, expected: list[Any], actual: list[Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Applies the warning declaration, if the case made one.

    Most cases do not, because checking everywhere would turn every pandas deprecation
    into a hundred failures that are all the same thing. The cases that do care are
    the ones where a warning is part of the contract, and the `NO_WARNING` form is the
    one that catches a library warning where pandas does not, which is what breaks
    somebody's `-W error` build.

    Args:
        case: The case.
        expected: What pandas warned.
        actual: What the subject warned.
        record: The result so far.

    Returns:
        The result.
    """
    if case.warns is None:
        return record
    wanted = None if case.warns == NO_WARNING else case.warns
    oracle_verdict = check_warnings(expected, wanted)
    if not oracle_verdict:
        record["outcome"] = FAIL
        record["detail"] = f"the case is wrong about pandas: {oracle_verdict.summary()}"
        return record
    verdict = check_warnings(actual, wanted)
    if not verdict:
        record["outcome"] = FAIL
        record["detail"] = verdict.summary()
    return record


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


def worker(engine_name: str, oracle: bool) -> int:
    """Runs cases named on stdin and writes one JSON line per event.

    Announcing a case before running it is what makes crash recovery possible. When
    this process dies, the parent knows the last announced case is the one that killed
    it, which no amount of exit code inspection would tell it.

    Args:
        engine_name: The subject engine.
        oracle: Whether the subject is pandas as well.

    Returns:
        A process exit code.
    """
    reference = load("pandas")
    subject = reference if oracle else load(engine_name)
    cases = registry()

    for line in sys.stdin:
        key = line.strip()
        if not key:
            continue
        case_id, frame_name = key.split("\t")
        print(json.dumps({"event": "start", "id": case_id, "frame": frame_name}), flush=True)
        record = run_case(cases[case_id], reference, subject, frame_name)
        print(json.dumps({"event": "result", "record": record}), flush=True)
    return 0


def _spawn(engine_name: str, oracle: bool) -> subprocess.Popen[str]:
    """Starts a worker.

    Args:
        engine_name: The subject engine.
        oracle: Whether to run pandas against pandas.

    Returns:
        The process.
    """
    command = [sys.executable, "-m", "fpcompat.runner", "--worker", "--engine", engine_name]
    if oracle:
        command.append("--oracle")
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )


def drive(work: list[tuple[str, str]], engine_name: str, oracle: bool) -> list[dict[str, Any]]:
    """Runs a work list in a worker, restarting it when it dies.

    Args:
        work: Case id and frame name pairs.
        engine_name: The subject engine.
        oracle: Whether to run pandas against pandas.

    Returns:
        One record per pair, in order.
    """
    records: list[dict[str, Any]] = []
    remaining = list(work)
    crashes = 0

    while remaining:
        process = _spawn(engine_name, oracle)
        assert process.stdin is not None and process.stdout is not None
        for case_id, frame_name in remaining:
            process.stdin.write(f"{case_id}\t{frame_name}\n")
        process.stdin.close()

        started: tuple[str, str] | None = None
        done = 0
        for line in process.stdout:
            event = json.loads(line)
            if event["event"] == "start":
                started = (event["id"], event["frame"])
                continue
            records.append(event["record"])
            started = None
            done += 1
        code = process.wait()

        if done == len(remaining) and started is None:
            return records

        crashes += 1
        # The worker died. Whatever it had announced and not finished is the case that
        # killed it, and that is a failure of the loudest kind rather than a skip.
        if started is not None:
            records.append(
                {
                    "id": started[0],
                    "frame": started[1],
                    "api": "",
                    "section": "",
                    "level": "",
                    "outcome": FAIL,
                    "detail": (
                        f"the worker died with exit code {code} while running this "
                        "case, which is a crash and is counted as a failure"
                    ),
                    "relaxations_used": [],
                    "seconds": 0.0,
                }
            )
            remaining = remaining[done + 1 :]
        else:
            remaining = remaining[done:]
        if crashes > 20:
            raise RuntimeError(
                "the worker died 20 times, which is not a conformance result any more, "
                "it is a broken environment"
            )
    return records


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(engine_name: str, oracle: bool, pattern: str | None) -> dict[str, Any]:
    """Runs a selection and builds the result document.

    Args:
        engine_name: The subject engine.
        oracle: Whether to run pandas against pandas.
        pattern: The case filter.

    Returns:
        The result document.
    """
    problems = corpus.compare(corpus.manifest(), json.loads(corpus.MANIFEST.read_text()))
    if problems:
        raise RuntimeError(
            "the corpus does not match its manifest, so no result computed against it "
            f"means anything: {problems[0]}"
        )

    cases = select(pattern)
    if not cases:
        # Zero of zero is not a hundred percent. A filter with a typo in it would
        # otherwise run nothing, count nothing, fail nothing and print a clean score,
        # which is the most misleading thing this program could do.
        raise RuntimeError(
            f"the filter {pattern!r} matches no cases, so there is nothing to run and "
            "nothing to conclude"
        )
    work = [(item.id, frame) for item in cases for frame in item.frames]
    started = time.perf_counter()
    records = drive(work, engine_name, oracle)

    totals = dict.fromkeys(OUTCOMES, 0)
    for record in records:
        totals[record["outcome"]] += 1

    versions = load("pandas").versions()
    if not oracle:
        versions |= load(engine_name).versions()

    return {
        "engine": "pandas" if oracle else engine_name,
        "oracle": oracle,
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 3),
        "versions": versions,
        "cases": len(cases),
        "runs": len(records),
        "totals": totals,
        "records": records,
    }


def report(document: dict[str, Any]) -> None:
    """Prints the summary line and every non pass.

    Args:
        document: The result document.
    """
    totals = document["totals"]
    print(
        f"{document['cases']} cases, {document['runs']} runs in {document['seconds']}s: "
        + ", ".join(f"{totals[name]} {name}" for name in OUTCOMES)
    )
    for record in document["records"]:
        if record["outcome"] != PASS:
            print(f"  {record['outcome']:>14}  {record['id']}  [{record['frame']}]")
            if record["detail"]:
                print(f"                  {record['detail']}")


def main(argv: list[str] | None = None) -> int:
    """Runs the suite.

    Args:
        argv: Command line arguments.

    Returns:
        A process exit code. Non zero when anything did not pass in oracle mode,
        because an oracle that is not perfect is a bug in this repository and no
        number it produces is publishable until it is fixed.
    """
    parser = argparse.ArgumentParser(description="Run the conformance cases.")
    parser.add_argument("--engine", default="pandas", help="pandas or firepanda")
    parser.add_argument("--oracle", action="store_true", help="run pandas against pandas")
    parser.add_argument("--filter", dest="pattern", help="a substring of the id, api or section")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", help="where to write the result file")
    args = parser.parse_args(argv)

    if args.worker:
        return worker(args.engine, args.oracle)

    document = run(args.engine, args.oracle, args.pattern)
    report(document)

    RESULTS.mkdir(exist_ok=True)
    target = Path(args.out) if args.out else RESULTS / f"{document['engine']}.json"
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target}")

    if args.oracle:
        wrong = document["runs"] - document["totals"][PASS]
        if wrong:
            print(
                f"\nthe oracle is not perfect: {wrong} of {document['runs']} runs did "
                "not pass. That is a bug in this repository rather than in anything it "
                "measures, and no result from this harness is publishable until it is "
                "fixed.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
