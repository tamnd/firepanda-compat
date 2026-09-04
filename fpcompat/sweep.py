"""Checks that every declared relaxation is load bearing.

Implements the check from `docs/specs/05-comparison.md`, and not the way that
document originally specified it, for a reason worth reading before changing
anything here.

A relaxation makes the comparison weaker. Two exist, `grouped_order` and
`row_order`, both closed and both required to carry a reason, and that is most of
what keeps the set honest. What it does not keep honest is a case that declares one
it does not need, because a declaration nobody needs never fails and therefore never
gets noticed. Over a few hundred cases that is how a strict suite turns into a loose
one, one reasonable looking line at a time.

The specified check was to run the oracle a second time with each relaxation
disabled and delete the ones whose cases still passed. That check cannot work. The
oracle is pandas against pandas: the same library, the same expression, the same
frame, so both sides come back in the same row order every time. Sorting both sides
or not sorting either changes nothing, so no case fails, so every relaxation reads
as unnecessary. Run over the registry as it stands that is 16 declarations out of
16, all of them deleted, and all of them needed the day an engine whose group by is
a hash aggregation runs the suite. A check that cannot fail deleting the thing it
was meant to protect is the exact failure this repository exists to avoid, arriving
from the one direction nobody watches.

So the relaxation is disabled against an adversary rather than against an identical
answer. Take pandas' own answer, permute its rows, and ask two questions of it.

With the relaxation declared, does the case still pass? It must. A relaxation that
does not survive a permutation of the answer is not delivering the order
insensitivity it claims, which is a bug in the comparison layer rather than in the
case.

With the relaxation disabled, does the case fail? It must. If a permuted answer
still compares equal without it, nothing about this case depends on it and the line
should go.

That runs today with only pandas installed, and it tests the property the
declaration actually asserts, which is that this answer means the same thing in a
different order. It holds whatever engine is on the other side, which the specified
version never could.

The permutation is a row reversal. It is deterministic, it needs no seed, and it is
a genuine permutation of any answer with two or more rows that is not the same read
backwards. When an answer cannot be permuted at all, because it is a scalar or has
one row on every frame the case runs on, that is reported rather than passed over: a
relaxation on an answer with no order is dead on every engine, present and future.

Tolerance classes are deliberately not checked here. A case declaring
`ACCUMULATION` is making a claim about how some other engine accumulates, and pandas
against pandas cannot produce evidence for or against it. Perturbing an answer by
the tolerance would test the tolerance arithmetic, which `test_compare.py` already
does. That one stays a review question.

Usage:
    python -m fpcompat.sweep
    python -m fpcompat.sweep --filter reshape --json results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from fpcompat.cases import Case, select
from fpcompat.compare import Rules, compare
from fpcompat.engines import load
from fpcompat.runner import run_expression

ROOT = Path(__file__).resolve().parent.parent

# Answers larger than this are left until last. Proving a relaxation is load bearing
# takes one frame, and there is no reason for that frame to be the ten million row
# self join when the same case also runs on a twenty two row one. The sort inside the
# comparison is a Python level sort of rendered tuples, which is a little under half a
# minute at ten million rows and instant at twenty two, so the ordering here is the
# difference between a sweep that runs in CI and one that does not. A case whose only
# frame is a large one still gets checked, at the end and at full price.
LARGE = 100_000

# The three things a declaration can turn out to be.
NEEDED = "needed"
UNNECESSARY = "unnecessary"
NO_ORDER = "no order"

# And the fourth, which is not about the declaration at all.
BROKEN = "broken"


def permute(answer: Any) -> Any:
    """Returns the same answer with its rows in a different order, or None.

    A reversal, because it needs no seed and is a permutation of anything that is
    not the same read backwards. The index travels with its rows, which is what a
    genuinely reordered answer looks like, except when the index is a default
    RangeIndex carrying no information, where it is rebuilt so that the permuted
    answer still has a default index. Without that the permuted side would carry
    63 down to 0 and the comparison would fail on the index rather than on the row
    order, which would make every relaxation look load bearing for the wrong reason.

    Args:
        answer: What the case returned.

    Returns:
        The permuted answer, or None when there is no order to permute.
    """
    if isinstance(answer, pd.DataFrame | pd.Series):
        if len(answer) < 2:
            return None
        flipped = answer.iloc[::-1]
        if _is_default_index(answer.index):
            flipped = flipped.reset_index(drop=True)
        return flipped
    if isinstance(answer, pd.Index):
        return answer[::-1] if len(answer) >= 2 else None
    return None


def _is_default_index(index: pd.Index) -> bool:
    """Whether an index is a plain 0 to n-1 RangeIndex.

    The same rule `compare` applies, restated here rather than imported, because the
    one in `compare` is private and this module has no business reaching into it.

    Args:
        index: The index.

    Returns:
        True when the index carries no information.
    """
    return (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and index.start == 0
        and index.step == 1
    )


def probe(case: Case, relaxation: str, expected: Any, permuted: Any) -> dict[str, Any]:
    """Asks the two questions of one declaration on one frame.

    Args:
        case: The case.
        relaxation: The declared relaxation being tested.
        expected: The answer pandas gave.
        permuted: The same answer with its rows moved.

    Returns:
        A verdict with a `state` of `needed`, `unnecessary` or `broken`.
    """
    with_it = compare(expected, permuted, case.rules)
    if not with_it.equal:
        return {
            "state": BROKEN,
            "detail": (
                "the case does not pass on a permutation of its own answer even with "
                f"{relaxation} declared, so the relaxation is not delivering the order "
                f"insensitivity the case is claiming: {with_it.summary()}"
            ),
        }
    without_it = compare(expected, permuted, case.rules.without(relaxation))
    if without_it.equal:
        return {
            "state": UNNECESSARY,
            "detail": (
                "a permutation of this answer compares equal without "
                f"{relaxation}, so nothing about this case depends on it"
            ),
        }
    return {"state": NEEDED, "detail": without_it.summary()}


def sweep_case(case: Case, engine: Any) -> dict[str, dict[str, Any]]:
    """Tests every relaxation one case declares.

    Small answers first, and each relaxation stops as soon as one frame proves it,
    because a declaration is load bearing if it is load bearing anywhere.

    Args:
        case: The case.
        engine: The pandas engine.

    Returns:
        A verdict per declared relaxation.
    """
    pending = sorted(case.rules.relaxations)
    verdicts: dict[str, dict[str, Any]] = {}
    state: dict[str, bool] = {"permutable": False}
    deferred: list[str] = []

    def attempt(frame_name: str, expected: Any) -> None:
        """Tests whatever is still unproven against one frame's answer."""
        permuted = permute(expected)
        if permuted is None:
            return
        if compare(expected, permuted, Rules()).equal:
            # Reversing it gave back something indistinguishable, so this frame has
            # nothing to say either way. An answer whose rows are all the same row
            # does this, and so does a single group.
            return
        state["permutable"] = True
        for relaxation in list(pending):
            verdict = probe(case, relaxation, expected, permuted) | {"frame": frame_name}
            if verdict["state"] == NEEDED:
                verdicts[relaxation] = verdict
                pending.remove(relaxation)
            else:
                verdicts.setdefault(relaxation, verdict)

    for frame_name in case.frames:
        if not pending:
            break
        expected, error, _ = run_expression(case, engine, frame_name)
        if error is not None:
            # The oracle reports this properly. Here it only means the sweep has no
            # answer to permute, and saying that is better than saying nothing.
            return {
                relaxation: {
                    "state": BROKEN,
                    "frame": frame_name,
                    "detail": (
                        f"the case raised {type(error).__name__}: {error}, so there is "
                        "no answer to permute. The oracle is where this gets diagnosed"
                    ),
                }
                for relaxation in sorted(case.rules.relaxations)
            }
        if _rows(expected) > LARGE and len(case.frames) > 1:
            deferred.append(frame_name)
            continue
        attempt(frame_name, expected)

    # The large answers, last, and only for whatever a small one did not settle.
    for frame_name in deferred:
        if not pending:
            break
        expected, error, _ = run_expression(case, engine, frame_name)
        if error is None:
            attempt(frame_name, expected)

    for relaxation in pending:
        verdicts.setdefault(
            relaxation,
            {
                "state": UNNECESSARY if state["permutable"] else NO_ORDER,
                "frame": "",
                "detail": (
                    f"a permutation of this answer compares equal without {relaxation} "
                    "on every frame this case runs on"
                )
                if state["permutable"]
                else (
                    "no frame this case runs on produced an answer whose rows can be "
                    "put in a different order, so this relaxation cannot matter on any "
                    "engine"
                ),
            },
        )
    return verdicts


def _rows(answer: Any) -> int:
    """How many rows an answer has, or zero when it does not have rows.

    Args:
        answer: The answer.

    Returns:
        The row count.
    """
    try:
        return len(answer)
    except TypeError:
        return 0


def sweep(pattern: str | None = None) -> dict[str, Any]:
    """Walks the registry and tests every relaxation in it.

    Args:
        pattern: A case filter, the same one the runner takes.

    Returns:
        The sweep document.
    """
    engine = load("pandas")
    cases = [item for item in select(pattern) if item.rules.relaxations]
    findings: dict[str, dict[str, Any]] = {}
    for case in cases:
        verdicts = sweep_case(case, engine)
        for relaxation, verdict in verdicts.items():
            findings[f"{case.id} {relaxation}"] = {
                "case": case.id,
                "relaxation": relaxation,
                "reason": case.rules.reason,
                **verdict,
            }
    counted = {
        state: sorted(key for key, value in findings.items() if value["state"] == state)
        for state in (NEEDED, UNNECESSARY, NO_ORDER, BROKEN)
    }
    return {
        "check": "relaxation sweep",
        "cases": len(cases),
        "declarations": len(findings),
        "findings": findings,
        "by_state": counted,
        "clean": not (counted[UNNECESSARY] or counted[NO_ORDER] or counted[BROKEN]),
    }


def render(document: dict[str, Any]) -> str:
    """Turns a sweep into something a person reads.

    Args:
        document: What `sweep` returned.

    Returns:
        The report.
    """
    if not document["declarations"]:
        return (
            "no case in this selection declares a relaxation, so there is nothing to "
            "sweep. That is a fact about the filter rather than a clean result."
        )

    lines = [
        f"{document['declarations']} relaxation declarations across "
        f"{document['cases']} cases, each tested against a permutation of its own "
        "answer",
        "",
    ]
    for key, finding in document["findings"].items():
        where = f" on {finding['frame']}" if finding["frame"] else ""
        lines.append(f"{finding['state']:12s} {key}{where}")

    for state, heading in (
        (
            BROKEN,
            "these do not pass a permutation of their own answer even with the "
            "relaxation declared, which is a bug in the comparison layer and not in the case",
        ),
        (
            UNNECESSARY,
            "these declare a relaxation nothing about the case depends on, and "
            "the fix is to delete the line",
        ),
        (
            NO_ORDER,
            "these declare an order relaxation on an answer that has no order to "
            "relax, so it is dead on every engine and the fix is to delete the line",
        ),
    ):
        listed = document["by_state"][state]
        if not listed:
            continue
        lines.extend(["", f"{len(listed)} {heading}:"])
        for key in listed:
            lines.append(f"    {key}: {document['findings'][key]['detail']}")

    lines.append("")
    if document["clean"]:
        lines.append(
            f"all {document['declarations']} declarations are load bearing: each one, "
            "disabled on its own, makes its case fail on a reordering of the answer "
            "pandas itself produced"
        )
    else:
        lines.append(
            "a relaxation that nothing depends on is how a strict suite becomes a "
            "loose one, one reasonable looking line at a time, so this exits non zero"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Runs the sweep from the command line.

    Args:
        argv: The arguments, or None for `sys.argv`.

    Returns:
        Zero when every declaration is load bearing, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default=None, help="a case id, api or section substring")
    parser.add_argument("--json", type=Path, default=None, help="also write the document here")
    args = parser.parse_args(argv)

    document = sweep(args.filter)
    print(render(document))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0 if document["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
