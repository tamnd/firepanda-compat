"""The divergence registry, which asserts rather than excuses.

Implements `docs/specs/06-divergences.md`.

A divergence is a place where firepanda deliberately does not do what pandas does.
Every one is a decision, every decision is written down before the case is allowed to
fail, and the entry is an assertion about what happens rather than a permission slip
for what does not.

The rule the whole file exists to enforce is that a registered divergence must
diverge. The runner does not skip a case that matches an entry. It runs it and it
requires the outcome the entry declares, so an entry saying `expect = "raises"` fails
loudly the day the case starts returning an answer, with a message saying the registry
is out of date rather than a message blaming the library. That inverts the usual
relationship. In most suites a known failure list is where cases go to be forgotten,
and it grows because adding an entry is cheaper than fixing a bug. Here an entry is an
assertion with a maintenance cost, and the day somebody implements backreferences by
switching the regex engine, the registry is what tells them to come and delete the
entry.

Nothing here applies in oracle mode. Both engines are pandas then, so there is nothing
for a statement about firepanda to be true of, and applying the registry would turn
every entry into a false failure. The divergence cases still run in oracle mode and
they still have to pass, which is what keeps them honest: a case whose pandas side is
broken cannot be evidence about firepanda either.

Usage:
    python -m fpcompat.divergences --list
    python -m fpcompat.divergences --page
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(__file__).resolve().parent / "divergences.toml"
PAGE = ROOT / "docs" / "divergences.md"

# `engine` is a deliberate design difference. `upstream` is pandas doing something we
# consider a bug, where we match the corrected behaviour, which is the rarest and most
# suspicious kind and needs a link to the pandas issue. `unsupported` is a scheduling
# fact rather than a design one and carries a milestone. `pending` is a difference with
# an expiry date, which is how a temporary divergence is allowed to exist without
# becoming permanent through inattention.
KINDS = ("engine", "upstream", "unsupported", "pending")

# What the entry claims happens instead. `raises` means the subject refuses the
# operation. `differs` means it answers, and the answer is not the pandas one.
EXPECTATIONS = ("raises", "differs")

# The shortest reason anybody has ever written that a user would accept is longer than
# this. The limit is not a quality check, it is a check that the field was filled in
# with a sentence rather than with a word.
MIN_REASON = 60

# Phrases that describe how hard the work is rather than why the difference exists.
# None of them is a reason for `kind = "engine"`, because the difference between a
# design and an excuse is exactly the difference between those two things.
NOT_A_REASON = (
    "hard to implement",
    "difficult to implement",
    "too slow to implement",
    "no time",
    "not implemented yet",
    "do this later",
    "maybe later",
)

ID_PATTERN = re.compile(r"^[a-z0-9]+(/[a-z0-9][a-z0-9._-]*)+$")


class DivergenceError(ValueError):
    """The registry is wrong.

    Always fatal and never an outcome, for the same reason a broken case declaration
    is fatal. A registry that does not load is a bug in this repository, and running
    the suite without it would silently turn every registered divergence into a
    failure of the library under test.
    """


@dataclass(frozen=True)
class Divergence:
    """One entry.

    Attributes:
        id: Stable, and it appears in the generated page users read.
        cases: Case id patterns. A pattern may cover a family of generated cases and
            may not cover a whole section.
        kind: One of `KINDS`.
        reason: Why the difference exists, in terms a user would accept.
        spec: Where the decision is written down.
        expect: What happens instead, one of `EXPECTATIONS`.
        instead: What firepanda produces, for a `differs` entry. Differs without
            saying how is not a specification, it is a shrug.
        since: When the entry was added.
        expires: For a `pending` entry, the date it stops being allowed to exist.
        milestone: For an `unsupported` entry, when it is scheduled.
        upstream: For an `upstream` entry, the pandas issue.
    """

    id: str
    cases: tuple[str, ...]
    kind: str
    reason: str
    spec: str
    expect: str
    instead: str = ""
    since: str = ""
    expires: str = ""
    milestone: str = ""
    upstream: str = ""

    def matches(self, case_id: str) -> bool:
        """Whether this entry covers a case.

        Args:
            case_id: The case id.

        Returns:
            Whether any pattern matches.
        """
        return any(fnmatch.fnmatchcase(case_id, pattern) for pattern in self.cases)

    def is_expired(self, today: date | None = None) -> bool:
        """Whether a pending entry has run out of time.

        Args:
            today: The date to compare against, for tests.

        Returns:
            Whether the entry expired.
        """
        if not self.expires:
            return False
        return date.fromisoformat(self.expires) < (today or date.today())


def _require(entry: dict[str, Any], field: str, why: str) -> str:
    """Reads a required string field.

    Args:
        entry: The raw table.
        field: The field name.
        why: What the field is for, for the message.

    Returns:
        The value.

    Raises:
        DivergenceError: When it is missing or empty.
    """
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DivergenceError(f"{entry.get('id', 'an entry')} has no {field}, and {why}")
    return value


def _check_patterns(entry: Divergence, known: dict[str, Any] | None) -> None:
    """Checks that the patterns point at real cases and are not blankets.

    A blanket entry is not a divergence, it is the score being redefined rather than
    measured, so a pattern that would cover an entire section is refused no matter how
    good the reason attached to it is.

    Args:
        entry: The entry.
        known: Case id to case, or None to skip the existence check.

    Raises:
        DivergenceError: When a pattern is a blanket, matches nothing, or covers a
            whole section.
    """
    for pattern in entry.cases:
        head = pattern.split("*", 1)[0]
        # The literal prefix has to name a section and one component under it, so
        # `divergences/plotting/*` is a family and `divergences/*` is a blanket.
        if len([part for part in head.split("/") if part]) < 2:
            raise DivergenceError(
                f"{entry.id} has the pattern {pattern!r}, which is a blanket. A "
                "pattern has to name a section and at least one component under it "
                "before its first wildcard, because at the point where an entry "
                "covers a whole section the score has been redefined rather than "
                "measured"
            )
    if known is None:
        return
    sections: dict[str, int] = {}
    for case_id in known:
        sections[case_id.split("/", 1)[0]] = sections.get(case_id.split("/", 1)[0], 0) + 1
    for pattern in entry.cases:
        hit = [case_id for case_id in known if fnmatch.fnmatchcase(case_id, pattern)]
        if not hit:
            raise DivergenceError(
                f"{entry.id} has the pattern {pattern!r}, which matches no case. An "
                "entry that asserts nothing about anything is not an assertion, and a "
                "pattern left behind by a renamed case is how a registry rots"
            )
        section = pattern.split("/", 1)[0]
        if len(hit) == sections.get(section, 0):
            raise DivergenceError(
                f"{entry.id} has the pattern {pattern!r}, which covers every case in "
                f"the {section} section. That is the score being redefined rather "
                "than measured"
            )


def parse(document: dict[str, Any], known: dict[str, Any] | None = None) -> list[Divergence]:
    """Validates a loaded registry.

    Args:
        document: The parsed TOML.
        known: Case id to case, for the pattern check, or None to skip it.

    Returns:
        The entries in file order.

    Raises:
        DivergenceError: When anything about the registry is wrong.
    """
    entries: list[Divergence] = []
    seen: set[str] = set()

    for raw in document.get("divergence", []):
        entry_id = _require(raw, "id", "an entry with no id cannot be referred to by anything")
        if not ID_PATTERN.match(entry_id):
            raise DivergenceError(
                f"{entry_id!r} is not a usable entry id, it has to match {ID_PATTERN}"
            )
        if entry_id in seen:
            raise DivergenceError(
                f"{entry_id} appears twice, and two entries cannot make one claim"
            )
        seen.add(entry_id)

        kind = _require(raw, "kind", f"an entry has to be one of {', '.join(KINDS)}")
        if kind not in KINDS:
            raise DivergenceError(
                f"{entry_id} has kind {kind}, and the kinds are {', '.join(KINDS)}"
            )

        expect = _require(raw, "expect", f"it has to be one of {', '.join(EXPECTATIONS)}")
        if expect not in EXPECTATIONS:
            raise DivergenceError(
                f"{entry_id} expects {expect}, and an entry either says the operation raises "
                "or says it answers differently. There is no third thing it could mean"
            )

        reason = _require(raw, "reason", "an unexplained divergence is an excuse")
        if len(reason) < MIN_REASON:
            raise DivergenceError(
                f"{entry_id} has a {len(reason)} character reason, which is a word rather "
                "than a sentence. This is the text a user reads when their program "
                "does something different, so it has to be worth reading"
            )
        lowered = reason.lower()
        # Whole phrases and not substrings. The first version of this check matched
        # "later" anywhere and rejected a perfectly good reason for the pickle entry
        # because it contained the words "a later pandas", which is the sort of thing
        # that teaches people to write worse reasons rather than better ones.
        if kind == "engine" and any(phrase in lowered for phrase in NOT_A_REASON):
            raise DivergenceError(
                f"{entry_id} is kind engine and its reason describes how hard the work is. "
                "That is not a design decision, it is kind unsupported with a "
                "milestone on it, and the difference between those two words is the "
                "difference between a design and an excuse"
            )

        spec = _require(raw, "spec", "a decision nobody wrote down is not a decision")
        since = _require(raw, "since", "an entry with no date cannot be aged")
        date.fromisoformat(since)

        cases = raw.get("cases")
        if not isinstance(cases, list) or not cases:
            raise DivergenceError(f"{entry_id} covers no cases, so it asserts nothing")

        instead = raw.get("instead", "")
        if expect == "differs" and not instead:
            raise DivergenceError(
                f"{entry_id} says the answer differs and does not say how. Differs without "
                "saying how is not a specification, and it is the entry a user reads "
                "when their program quietly returned the wrong number"
            )

        expires = raw.get("expires", "")
        if kind == "pending" and not expires:
            raise DivergenceError(
                f"{entry_id} is kind pending and has no expires date. Pending without a date "
                "is permanent, and temporarily is the most load bearing word in "
                "software"
            )
        if expires:
            date.fromisoformat(expires)

        milestone = raw.get("milestone", "")
        if kind == "unsupported" and not milestone:
            raise DivergenceError(
                f"{entry_id} is kind unsupported and names no milestone. Unsupported is a "
                "scheduling fact, and a schedule with no date on it is a wish"
            )

        upstream = raw.get("upstream", "")
        if kind == "upstream" and not upstream:
            raise DivergenceError(
                f"{entry_id} is kind upstream and links to no pandas issue. Claiming pandas "
                "is wrong is the strongest thing this registry can say and it does not "
                "get to be said without a citation"
            )

        entry = Divergence(
            id=entry_id,
            cases=tuple(cases),
            kind=kind,
            reason=reason,
            spec=spec,
            expect=expect,
            instead=instead,
            since=since,
            expires=expires,
            milestone=milestone,
            upstream=upstream,
        )
        _check_patterns(entry, known)
        entries.append(entry)

    return entries


@cache
def registry() -> tuple[Divergence, ...]:
    """The validated registry, with expiry enforced.

    An expired pending entry stops the whole run rather than failing one case, and it
    does so even in oracle mode. That is deliberate and it is the only mechanism in
    this repository with a calendar in it. A date that has passed means somebody
    promised a difference was temporary and the promise came due, and finding that out
    from a red build is the entire point of writing the date down.

    Returns:
        The entries.

    Raises:
        DivergenceError: When the registry is invalid or an entry has expired.
    """
    from fpcompat.cases import registry as cases

    if not REGISTRY.exists():
        return ()
    entries = parse(tomllib.loads(REGISTRY.read_text()), cases())
    expired = [entry for entry in entries if entry.is_expired()]
    if expired:
        raise DivergenceError(
            f"{expired[0].id} expired on {expired[0].expires} and is still in the "
            "registry. A pending divergence that outlives its date is a permanent one "
            "that nobody decided on, so the run stops here until somebody either "
            "fixes it or argues for it in writing"
        )
    return tuple(entries)


def match(case_id: str) -> Divergence | None:
    """The entry covering a case, if there is one.

    Args:
        case_id: The case id.

    Returns:
        The entry or None.
    """
    for entry in registry():
        if entry.matches(case_id):
            return entry
    return None


def case_ids() -> frozenset[str]:
    """Every registered case id, for the runner and the scoreboard.

    Returns:
        The ids.
    """
    from fpcompat.cases import registry as cases

    return frozenset(case_id for case_id in cases() if match(case_id) is not None)


# ---------------------------------------------------------------------------
# The generated page
# ---------------------------------------------------------------------------


def page() -> str:
    """Builds the public divergence page.

    Generated rather than written, so it cannot fall behind the code. This is the
    honest version of a migration guide: it is the list of things a program that
    works today will do differently tomorrow, and a user is entitled to read it
    before they port anything rather than after.

    Returns:
        The markdown.
    """
    entries = registry()
    covered = case_ids()
    lines = [
        "# Divergences",
        "",
        "Generated by `python -m fpcompat.divergences --page`. Do not edit this file, "
        "edit `fpcompat/divergences.toml` and regenerate, which CI checks.",
        "",
        "Every entry here is a place where firepanda deliberately does not do what "
        "pandas does. None of them is a bug and none of them is a gap in the "
        "schedule, both of which are counted elsewhere. Each one is asserted by cases "
        "that run on every commit, so an entry that stopped being true fails the "
        "build rather than sitting here misleading somebody.",
        "",
        f"{len(entries)} entries, asserted by {len(covered)} cases.",
        "",
        "| entry | kind | what happens instead |",
        "| --- | --- | --- |",
    ]
    for entry in entries:
        instead = "the operation raises" if entry.expect == "raises" else entry.instead
        lines.append(f"| [{entry.id}](#{entry.id.replace('/', '')}) | {entry.kind} | {instead} |")
    lines.append("")

    for entry in entries:
        lines.append(f"## {entry.id}")
        lines.append("")
        lines.append(entry.reason)
        lines.append("")
        if entry.expect == "differs":
            lines.append(f"**Instead.** {entry.instead}")
            lines.append("")
        details = [f"Kind `{entry.kind}`", f"since {entry.since}", f"specified in `{entry.spec}`"]
        if entry.milestone:
            details.append(f"scheduled for {entry.milestone}")
        if entry.upstream:
            details.append(f"pandas issue {entry.upstream}")
        if entry.expires:
            details.append(f"expires {entry.expires}")
        lines.append(", ".join(details) + ".")
        lines.append("")
        lines.append("Asserted by:")
        lines.append("")
        for pattern in entry.cases:
            lines.append(f"- `{pattern}`")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Lists the registry or writes the page.

    Args:
        argv: Command line arguments.

    Returns:
        A process exit code. Non zero when `--check` finds the page stale.
    """
    parser = argparse.ArgumentParser(description="The divergence registry.")
    parser.add_argument("--list", action="store_true", help="print the entries")
    parser.add_argument("--page", action="store_true", help="write docs/divergences.md")
    parser.add_argument("--check", action="store_true", help="fail when the page is stale")
    args = parser.parse_args(argv)

    if args.check:
        current = PAGE.read_text() if PAGE.exists() else ""
        if current != page():
            print(
                "docs/divergences.md is stale. Run `pixi run divergences` and commit "
                "the result, because a generated page that drifts is worse than no "
                "page at all",
                file=sys.stderr,
            )
            return 1
        print(f"docs/divergences.md is current, {len(registry())} entries")
        return 0

    if args.page:
        PAGE.parent.mkdir(parents=True, exist_ok=True)
        PAGE.write_text(page())
        print(f"wrote {PAGE}, {len(registry())} entries")
        return 0

    for entry in registry():
        print(f"{entry.id:<34} {entry.kind:<12} {entry.expect:<8} {len(entry.cases)} patterns")
    print(f"{len(registry())} entries covering {len(case_ids())} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
