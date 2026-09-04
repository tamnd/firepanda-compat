"""The scoreboard.

Implements `docs/specs/07-scoreboard.md`.

Reads a result file and prints the summary line, the section table and the parameter
coverage table. It imports no pandas and it never runs a case. Everything it needs is
in the result file and in the committed inventory for the pandas version that file
names, which is what lets somebody render a run they did not produce.

Five rules from document 07 are enforced here rather than described.

**The denominator is the pandas surface.** Every percentage is over the callables
pandas has, not over the cases this repository wrote. A suite reporting a pass rate
over its own cases is reporting how good it is at writing cases it passes.

**An untouched parameter is a hole and not a pass.** The coverage table sits next to
the score and neither number is quotable without the other. A hundred percent L3 over
forty percent of the parameter space is a suite that is not finished, and the front
page has to say both numbers in the same breath.

**A divergence is displayed and not subtracted.** It has its own column. It never
moves a name into the passing count and it never moves it into the failing one.

**The run that lost is published.** The report prints whatever it read. There is no
threshold below which it declines to print, and the only gate is a ratchet.

**The oracle stands in front of the score.** If the pandas against pandas run was not
perfect, that is what the front page says, in place of the number, because a harness
that disagrees with itself has no business publishing a number about anything else.

Usage:
    python -m fpcompat.report                       # the score
    python -m fpcompat.report --coverage            # the work list
    python -m fpcompat.report --site site           # the three pages
    python -m fpcompat.report --ratchet             # fail if a section went backwards
    python -m fpcompat.report --ratchet --update    # record a new floor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fpcompat import sections

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RATCHET = ROOT / "ratchet.json"

# The levels the score is cumulative over. L4 is about which exception comes out and it
# is reported on its own, because a name with no failure mode to speak of is not
# behind on anything by lacking an L4 case.
LADDER = ("L0", "L1", "L2", "L3")


@dataclass
class Name:
    """What one pandas callable's cases said about it.

    Attributes:
        api: The pandas name.
        section: Its parity section.
        levels: Level to the outcomes seen at that level.
        covered: Parameters some case exercised.
        divergent: Whether any case on this name is a registered divergence.
    """

    api: str
    section: str
    levels: dict[str, set[str]] = field(default_factory=dict)
    covered: set[str] = field(default_factory=set)
    divergent: bool = False

    def attained(self) -> str | None:
        """The highest level this name reached.

        A name reaches a level when it has a passing case at that level and nothing
        at or below it failed or came back unimplemented. Levels with no case do not
        block, which means a name with a passing L2 case and no L1 case is reported at
        L2 and therefore counted at L1 as well. That inference is worth naming: it says
        producing the right default answer required the call to be accepted, which is
        true and is weaker than an L1 case, since it only speaks for the default
        arguments. Since the surface sweep landed, every callable has a real L0 case
        and every introspectable one has a real L1 case, so the inference almost never
        has to be made.

        Returns:
            The level name, or None when the name has no passing case at all.
        """
        reached = None
        for level in LADDER:
            outcomes = self.levels.get(level)
            if outcomes and outcomes - {"pass"}:
                # Something at this level is a failure, an unimplemented name or a
                # divergence. A divergence stops the climb too: the name does not do
                # what pandas does, which is the honest reading of a divergence, and
                # it is displayed in its own column rather than counted as a pass.
                return reached
            if outcomes:
                reached = level
        return reached

    def failed(self) -> bool:
        """Whether any case on this name failed."""
        return any("fail" in outcomes for outcomes in self.levels.values())

    def unimplemented(self) -> bool:
        """Whether any case on this name found it missing."""
        return any("unimplemented" in outcomes for outcomes in self.levels.values())


def read(path: Path) -> dict[str, Any]:
    """Loads a result file.

    Args:
        path: The file the runner wrote.

    Returns:
        The document.

    Raises:
        FileNotFoundError: With the command that produces one, since the most common
            reason for this is that nobody has run the suite yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no result file at {path}. Run `pixi run oracle` for the pandas against "
            "pandas run, or `pixi run conformance` for the real one"
        )
    return json.loads(path.read_text())


def collect(document: dict[str, Any]) -> dict[str, Name]:
    """Folds the records into one entry per pandas callable.

    Every callable in the inventory gets an entry, including the ones no case names,
    because a name nobody tested is the thing the coverage table exists to show and
    dropping it here would be the exact green washing document 07 forbids.

    Args:
        document: The result file.

    Returns:
        Qualified pandas name to what its cases said.
    """
    version = document["versions"]["pandas"]
    known = sections.callables(version)
    names = {api: Name(api, section) for api, section in known.items()}
    declarations = document.get("declarations", {})

    for record in document["records"]:
        api = record["api"]
        entry = names.get(api)
        if entry is None:
            # A case on a property or on an operator form. Both are real cases and
            # neither is a callable in the denominator, so they are evidence that goes
            # in the detail listing rather than in the section table.
            continue
        entry.levels.setdefault(record["level"], set()).add(record["outcome"])
        if record.get("divergence"):
            entry.divergent = True
        entry.covered.update(declarations.get(record["id"], {}).get("covers", ()))
    return names


def score(document: dict[str, Any]) -> dict[str, Any]:
    """Computes everything the report prints.

    Args:
        document: The result file.

    Returns:
        The totals, the per section rows and the coverage numbers.
    """
    version = document["versions"]["pandas"]
    names = collect(document)
    denominator = sections.denominator(version)
    parameters = sections.parameters(version)

    rows = {}
    for section in sections.SECTIONS:
        rows[section] = {
            "callables": denominator[section],
            "levels": dict.fromkeys(LADDER, 0),
            "divergent": 0,
            "fail": 0,
            "unimplemented": 0,
            "untested": 0,
            "parameters": 0,
            "covered": 0,
        }

    for entry in names.values():
        row = rows[entry.section]
        attained = entry.attained()
        if attained is not None:
            for level in LADDER:
                row["levels"][level] += 1
                if level == attained:
                    break
        if entry.divergent:
            row["divergent"] += 1
        if entry.failed():
            row["fail"] += 1
        if entry.unimplemented():
            row["unimplemented"] += 1
        if not entry.levels:
            row["untested"] += 1
        params = parameters.get(entry.api, ())
        row["parameters"] += len(params)
        row["covered"] += len(entry.covered & set(params))

    totals = {
        "callables": sum(row["callables"] for row in rows.values()),
        "levels": {level: sum(row["levels"][level] for row in rows.values()) for level in LADDER},
        "divergent": sum(row["divergent"] for row in rows.values()),
        "fail": sum(row["fail"] for row in rows.values()),
        "unimplemented": sum(row["unimplemented"] for row in rows.values()),
        "untested": sum(row["untested"] for row in rows.values()),
        "parameters": sum(row["parameters"] for row in rows.values()),
        "covered": sum(row["covered"] for row in rows.values()),
    }
    return {"rows": rows, "totals": totals, "names": names}


def percent(part: int, whole: int) -> str:
    """Renders a percentage, or a dash when there is nothing to divide by.

    Args:
        part: The numerator.
        whole: The denominator.

    Returns:
        A string like `27.7%`.
    """
    if not whole:
        return "  -  "
    return f"{100.0 * part / whole:.1f}%"


def summary(document: dict[str, Any], computed: dict[str, Any]) -> str:
    """The two line summary that goes in a commit message and a milestone comment.

    Args:
        document: The result file.
        computed: The output of `score`.

    Returns:
        Two lines.
    """
    totals = computed["totals"]
    versions = document["versions"]
    subject = document["engine"]
    subject_version = versions.get(subject, versions.get("firepanda", "unknown"))
    if document["oracle"]:
        subject, subject_version = "pandas", versions["pandas"]

    levels = totals["levels"]
    head = (
        f"{subject} {subject_version} vs pandas {versions['pandas']}   "
        f"L3 {levels['L3']}/{totals['callables']} ({percent(levels['L3'], totals['callables'])})   "
        f"L2 {levels['L2']}   L1 {levels['L1']}   L0 {levels['L0']}"
    )
    tail = (
        f"divergent {totals['divergent']}   unimplemented {totals['unimplemented']}   "
        f"fail {totals['fail']}   untested {totals['untested']}   "
        f"parameters {percent(totals['covered'], totals['parameters'])}   "
        f"cases {document['cases']} in {document['seconds']}s"
    )
    return f"{head}\n{tail}"


def oracle_warning(document: dict[str, Any]) -> str:
    """What stands in front of the score when the harness disagreed with itself.

    Args:
        document: The result file.

    Returns:
        The warning, or an empty string when the run was clean.
    """
    if not document["oracle"]:
        return ""
    wrong = document["runs"] - document["totals"]["pass"]
    if not wrong:
        return ""
    return (
        f"THE ORACLE IS NOT PERFECT: {wrong} of {document['runs']} pandas against "
        "pandas runs did not pass. That is a bug in the harness and no number below "
        "this line means anything until it is fixed."
    )


def section_table(computed: dict[str, Any]) -> str:
    """The per section table, ordered by how far behind each section is.

    A section at 90 percent is printed as 90 percent and not as done, which is the one
    sentence document 07 repeats from document 06, because this is the file where
    breaking it would be convenient.

    Args:
        computed: The output of `score`.

    Returns:
        A markdown table.
    """
    header = (
        "| Section | Callables | L3 | % | L2 | L1 | L0 | Divergent | Fail | "
        "Unimplemented | Untested | Parameters |"
    )
    lines = [header, "|" + "---|" * 12]

    def behind(item: tuple[str, dict[str, Any]]) -> tuple[float, int]:
        """Worst L3 rate first, and the bigger section first when two rates tie."""
        row = item[1]
        return (row["levels"]["L3"] / (row["callables"] or 1), -row["callables"])

    order = sorted(computed["rows"].items(), key=behind)
    for name, row in order:
        levels = row["levels"]
        lines.append(
            f"| {name} | {row['callables']} | {levels['L3']} | "
            f"{percent(levels['L3'], row['callables'])} | {levels['L2']} | {levels['L1']} | "
            f"{levels['L0']} | {row['divergent']} | {row['fail']} | {row['unimplemented']} | "
            f"{row['untested']} | {percent(row['covered'], row['parameters'])} |"
        )
    totals = computed["totals"]
    levels = totals["levels"]
    lines.append(
        f"| **all** | {totals['callables']} | {levels['L3']} | "
        f"{percent(levels['L3'], totals['callables'])} | {levels['L2']} | {levels['L1']} | "
        f"{levels['L0']} | {totals['divergent']} | {totals['fail']} | "
        f"{totals['unimplemented']} | {totals['untested']} | "
        f"{percent(totals['covered'], totals['parameters'])} |"
    )
    return "\n".join(lines)


def coverage_lines(document: dict[str, Any], computed: dict[str, Any]) -> list[str]:
    """One line per callable that has a parameter no case has ever touched.

    This is the work list. It is per parameter and not per callable on purpose: a
    callable with one of its nine parameters exercised is not covered, and reporting it
    as covered is how a suite comes to believe it has tested something.

    Args:
        document: The result file.
        computed: The output of `score`.

    Returns:
        The lines, worst first.
    """
    parameters = sections.parameters(document["versions"]["pandas"])
    rows = []
    for api, entry in computed["names"].items():
        params = parameters.get(api, ())
        if not params:
            continue
        uncovered = [name for name in params if name not in entry.covered]
        if not uncovered:
            continue
        rows.append((len(uncovered), api, entry.covered & set(params), uncovered))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [
        f"{api:<40} covered: {', '.join(sorted(covered)) or 'nothing':<40} "
        f"uncovered: {', '.join(uncovered)}"
        for _, api, covered, uncovered in rows
    ]


def unmeasurable(document: dict[str, Any]) -> list[str]:
    """The callables whose signature pandas itself cannot report.

    They have no L1 case and they never will, because both engines failing to
    introspect is not evidence of agreement. Listing them is the honest alternative to
    counting them.

    Args:
        document: The result file.

    Returns:
        The qualified names, sorted.
    """
    inventory = sections.inventory(document["versions"]["pandas"])
    found = []
    for space, entry in inventory["namespaces"].items():
        for member, info in entry["members"].items():
            if info["kind"] == "callable" and info.get("signature") is None:
                found.append(f"{space}.{member}")
    return sorted(found)


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------


def floors(computed: dict[str, Any]) -> dict[str, Any]:
    """The numbers the ratchet remembers.

    Args:
        computed: The output of `score`.

    Returns:
        Per section L3 counts and the total failure count.
    """
    return {
        "l3": {name: row["levels"]["L3"] for name, row in computed["rows"].items()},
        "fail": computed["totals"]["fail"],
    }


def check_ratchet(computed: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Compares a run against the committed floor.

    A ratchet and not a threshold. A threshold on a young project is either so low it
    means nothing or so high it blocks the work that would raise it, and either way
    people learn to route around it. What this refuses is going backwards.

    Args:
        computed: The output of `score`.
        recorded: The committed floor.

    Returns:
        One message per regression, empty when nothing went backwards.
    """
    problems = []
    current = floors(computed)
    for name, floor in recorded.get("l3", {}).items():
        now = current["l3"].get(name, 0)
        if now < floor:
            problems.append(
                f"{name} went from L3 {floor} to {now}. Raising the floor is part of "
                "the pull request that earns it and lowering it is a reviewable diff, "
                "so if this regression is deliberate, say why in the body under a "
                "Conformance: line and edit ratchet.json in the same commit"
            )
    if current["fail"] > recorded.get("fail", 0):
        problems.append(
            f"failures went from {recorded.get('fail', 0)} to {current['fail']}. "
            "Unimplemented is a schedule and a failure is a bug, and the bug count is "
            "the one that does not get to go up"
        )
    return problems


# ---------------------------------------------------------------------------
# The site
# ---------------------------------------------------------------------------


def front_page(document: dict[str, Any], computed: dict[str, Any]) -> str:
    """The score page.

    Args:
        document: The result file.
        computed: The output of `score`.

    Returns:
        Markdown.
    """
    versions = " ".join(f"{key} {value}" for key, value in sorted(document["versions"].items()))
    warning = oracle_warning(document)
    parts = [
        "# Conformance",
        "",
        f"Run at {document['when']}, {versions}.",
        "",
    ]
    if warning:
        parts += [f"> **{warning}**", ""]
    parts += [
        "```",
        summary(document, computed),
        "```",
        "",
        "The two numbers on the right of the second line are the ones that keep the "
        "first line honest. `untested` counts pandas callables no case in this suite "
        "has ever named, and `parameters` is the share of the pandas parameter space "
        "any case has exercised. A high L3 over a low parameter coverage is a suite "
        "that is not finished, so neither number is quotable without the other.",
        "",
        "## Per section",
        "",
        section_table(computed),
        "",
        "A section at 90 percent is 90 percent and not done.",
        "",
        "[Parameter coverage](coverage.md) and [the divergence list](divergences.md).",
    ]
    return "\n".join(parts) + "\n"


def coverage_page(document: dict[str, Any], computed: dict[str, Any]) -> str:
    """The coverage page.

    Args:
        document: The result file.
        computed: The output of `score`.

    Returns:
        Markdown.
    """
    totals = computed["totals"]
    lines = coverage_lines(document, computed)
    missing = unmeasurable(document)
    parts = [
        "# Parameter coverage",
        "",
        f"{totals['covered']} of {totals['parameters']} pandas parameters "
        f"({percent(totals['covered'], totals['parameters'])}) have been exercised by "
        "at least one case.",
        "",
        "An uncovered parameter is not a failure and it is not a pass. It is a hole in "
        "the measurement, and this page is the work list that closes it.",
        "",
        f"## {len(lines)} callables with an untouched parameter",
        "",
        "```",
        *lines,
        "```",
        "",
        f"## {len(missing)} callables with no readable signature",
        "",
        "`inspect.signature` refuses these, so pandas cannot report its own parameter "
        "list for them and no L1 case exists. Both engines failing to introspect is "
        "not evidence of agreement, so they are listed here rather than counted "
        "anywhere.",
        "",
        "```",
        *missing,
        "```",
    ]
    return "\n".join(parts) + "\n"


def write_site(document: dict[str, Any], computed: dict[str, Any], target: Path) -> list[Path]:
    """Writes the three pages.

    Args:
        document: The result file.
        computed: The output of `score`.
        target: The directory to write into.

    The divergence page is copied from `docs/divergences.md` rather than regenerated
    here. Regenerating it would mean importing the case registry, which imports pandas
    and pyarrow, and that would cost this module the one property it is built around:
    that it can render a run on a machine with neither installed. The committed page is
    generated by `pixi run divergences` and CI fails when it has drifted, so copying it
    is not a weaker guarantee, it is the same guarantee enforced somewhere else.

    Args:
        document: The result file.
        computed: The output of `score`.
        target: The directory to write into.

    Returns:
        The paths written.

    Raises:
        FileNotFoundError: When the divergence page has not been generated.
    """
    page = ROOT / "docs" / "divergences.md"
    if not page.exists():
        raise FileNotFoundError(f"no {page}. Run `pixi run divergences` to generate it")

    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (
        ("index.md", front_page(document, computed)),
        ("coverage.md", coverage_page(document, computed)),
        ("divergences.md", page.read_text()),
    ):
        path = target / name
        path.write_text(text)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Renders a result file.

    Args:
        argv: Command line arguments.

    Returns:
        A process exit code. Non zero when the ratchet was asked about and something
        went backwards, and non zero when the oracle run it was handed was not perfect.
    """
    parser = argparse.ArgumentParser(description="Print the conformance scoreboard.")
    parser.add_argument("--results", type=Path, help="the result file, default results/pandas.json")
    parser.add_argument("--coverage", action="store_true", help="the parameter work list")
    parser.add_argument("--site", type=Path, help="write the three pages into this directory")
    parser.add_argument("--ratchet", action="store_true", help="fail if a section went backwards")
    parser.add_argument("--update", action="store_true", help="record a new floor")
    args = parser.parse_args(argv)

    document = read(args.results or RESULTS / "pandas.json")
    computed = score(document)

    if args.ratchet:
        if args.update:
            RATCHET.write_text(json.dumps(floors(computed), indent=2, sort_keys=True) + "\n")
            print(f"wrote {RATCHET.relative_to(ROOT)}")
            return 0
        if not RATCHET.exists():
            print(f"no {RATCHET.name}: run with --ratchet --update and commit it")
            return 1
        problems = check_ratchet(computed, json.loads(RATCHET.read_text()))
        for problem in problems:
            print(problem)
        if problems:
            return 1
        print("nothing went backwards")
        return 0

    if args.site:
        for path in write_site(document, computed, args.site):
            print(f"wrote {path}")
        return 0

    warning = oracle_warning(document)
    if warning:
        print(warning)
        print()

    if args.coverage:
        for line in coverage_lines(document, computed):
            print(line)
        totals = computed["totals"]
        print(
            f"\n{totals['covered']} of {totals['parameters']} parameters "
            f"({percent(totals['covered'], totals['parameters'])}) exercised by at "
            "least one case"
        )
        return 1 if warning else 0

    print(summary(document, computed))
    print()
    print(section_table(computed))
    return 1 if warning else 0


if __name__ == "__main__":
    raise SystemExit(main())
