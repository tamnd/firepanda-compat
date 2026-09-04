"""Tests for the divergence registry.

The test this file exists for is `test_an_entry_that_stopped_diverging_fails`. Every
other suite in the world treats a known failure list as a place where cases go to be
forgotten, and the only thing that stops this one becoming that is the assertion that
a registered divergence must actually diverge. If that assertion ever quietly stops
holding, this repository turns into a machine for explaining away failures, and the
explanations will all be well written.

The rest is the loader refusing bad entries. A blanket pattern, an entry pointing at
no case, a reason that describes how hard the work is, a pending entry with no expiry
date. All fatal, because every one of them is the score being redefined rather than
measured.
"""

from __future__ import annotations

import tomllib
from datetime import date, timedelta

import pandas as pd
import pytest

from fpcompat import divergences, runner
from fpcompat.cases import Case, registry
from fpcompat.compare import Rules
from fpcompat.divergences import Divergence, DivergenceError, parse
from fpcompat.engines import load

ORACLE = load("pandas")

GOOD = {
    "id": "engine/scratch",
    "cases": ["divergences/plotting/frame-plot"],
    "kind": "engine",
    "reason": "A reason long enough to be a sentence, which is the point of the length "
    "check, since the field is what a user reads when their program behaves differently.",
    "spec": "06-pandas-parity.md#somewhere",
    "expect": "raises",
    "since": "2026-09-04",
}


def load_one(**overrides):
    """Parses a registry of exactly one entry, with the fields a test wants changed."""
    entry = dict(GOOD)
    entry.update(overrides)
    return parse({"divergence": [entry]}, registry())


# ---------------------------------------------------------------------------
# The rule that keeps the registry honest
# ---------------------------------------------------------------------------


def build(**overrides) -> Case:
    fields = {
        "id": "divergences/plotting/frame-plot",
        "api": "DataFrame.plot",
        "section": "divergences",
        "milestone": "M6",
        "level": "L0",
        "covers": (),
        "frames": ("two",),
        "expr": lambda pd, df: type(df.plot).__name__,
        "rules": Rules(),
    }
    fields.update(overrides)
    return Case(**fields)


class Subject:
    """An engine that is pandas, standing in for firepanda.

    The lie is in the case expression rather than in here, because what a divergence
    test needs to vary is what the second call returns and not what the engine is.
    """

    def module(self):
        return pd

    def frame(self, name):
        return ORACLE.frame(name)

    def versions(self):
        return ORACLE.versions()


def test_an_entry_that_stopped_diverging_fails(monkeypatch):
    """The most important test in this repository.

    The entry says the operation raises. The subject returns the pandas answer. That
    is the day somebody implemented the feature and forgot to delete the entry, and it
    has to be loud, and the message has to blame the registry rather than the library
    because on that day the library got better.
    """
    entry = load_one()[0]
    monkeypatch.setattr(runner, "divergence_for", lambda case_id: entry)
    record = runner.run_case(build(), ORACLE, Subject(), "two")
    assert record["outcome"] == runner.FAIL
    assert "out of date" in record["detail"]
    assert record["divergence"] == "engine/scratch"


def test_an_entry_that_is_still_diverging_is_divergent(monkeypatch):
    entry = load_one()[0]
    monkeypatch.setattr(runner, "divergence_for", lambda case_id: entry)

    def raising(module, df):
        raise TypeError("firepanda has no plotting, use to_pandas()")

    record = runner.run_case(build(expr=_subject_only(raising)), ORACLE, Subject(), "two")
    assert record["outcome"] == runner.DIVERGENT
    assert "to_pandas" in record["detail"]


def _subject_only(replacement):
    """An expression that behaves normally for the oracle and differently after."""
    seen = {"count": 0}

    def expr(module, df):
        seen["count"] += 1
        if seen["count"] == 1:
            return type(df.plot).__name__
        return replacement(module, df)

    return expr


def test_a_differs_entry_that_matched_pandas_fails(monkeypatch):
    entry = load_one(expect="differs", instead="it is positional")[0]
    monkeypatch.setattr(runner, "divergence_for", lambda case_id: entry)
    record = runner.run_case(build(), ORACLE, Subject(), "two")
    assert record["outcome"] == runner.FAIL
    assert "matched pandas exactly" in record["detail"]


def test_a_differs_entry_that_raised_fails(monkeypatch):
    """Refusing is a different promise from answering differently."""
    entry = load_one(expect="differs", instead="it is positional")[0]
    monkeypatch.setattr(runner, "divergence_for", lambda case_id: entry)

    def raising(module, df):
        raise TypeError("no")

    record = runner.run_case(build(expr=_subject_only(raising)), ORACLE, Subject(), "two")
    assert record["outcome"] == runner.FAIL
    assert "expect = raises" in record["detail"]


def test_a_differs_entry_that_differs_is_divergent(monkeypatch):
    entry = load_one(expect="differs", instead="it is positional")[0]
    monkeypatch.setattr(runner, "divergence_for", lambda case_id: entry)
    record = runner.run_case(
        build(expr=_subject_only(lambda module, df: "SomethingElse")),
        ORACLE,
        Subject(),
        "two",
    )
    assert record["outcome"] == runner.DIVERGENT


def test_the_registry_does_not_apply_in_oracle_mode():
    """Both engines are pandas, so there is nothing for a claim about firepanda to be
    true of, and the divergence cases have to pass as ordinary cases."""
    record = runner.run_case(build(), ORACLE, ORACLE, "two", oracle_mode=True)
    assert record["outcome"] == runner.PASS
    assert record["divergence"] == ""


# ---------------------------------------------------------------------------
# What the loader refuses
# ---------------------------------------------------------------------------


def test_a_blanket_pattern_is_refused():
    """`divergences/*` redefines the score rather than measuring it."""
    with pytest.raises(DivergenceError, match="which is a blanket"):
        load_one(cases=["divergences/*"])


def test_a_pattern_covering_a_whole_section_is_refused():
    """Not a blanket by spelling, and a blanket by effect, which counts the same.

    The pattern names a section and a component under it, so it passes the shape
    check, and it still happens to match every case in that section. The registry
    refuses it anyway, because what makes a blanket a blanket is what it covers rather
    than how it is written.
    """
    known = {"tiny/thing/one": None, "tiny/thing/two": None}
    with pytest.raises(DivergenceError, match="covers every case"):
        parse({"divergence": [dict(GOOD, cases=["tiny/thing/*"])]}, known)


def test_a_section_wildcard_is_refused_as_a_blanket():
    with pytest.raises(DivergenceError, match="which is a blanket"):
        load_one(cases=["errors/*"])


def test_a_pattern_matching_nothing_is_refused():
    """A pattern left behind by a renamed case is how a registry rots."""
    with pytest.raises(DivergenceError, match="matches no case"):
        load_one(cases=["divergences/plotting/nothing-called-this"])


def test_an_unknown_kind_is_refused():
    with pytest.raises(DivergenceError, match="and the kinds are"):
        load_one(kind="probably")


def test_an_unknown_expectation_is_refused():
    with pytest.raises(DivergenceError, match="no third thing"):
        load_one(expect="sometimes")


def test_a_one_word_reason_is_refused():
    with pytest.raises(DivergenceError, match="rather than a sentence"):
        load_one(reason="performance")


def test_a_reason_about_difficulty_is_refused():
    """The difference between a design and an excuse is the words in this field."""
    with pytest.raises(DivergenceError, match="difference between a design and an excuse"):
        load_one(
            reason="This one is hard to implement and we would rather spend the effort "
            "somewhere else in the library where it does more good for more people."
        )


def test_a_difficulty_reason_is_allowed_for_unsupported():
    """Which is the whole point of having the word unsupported."""
    entries = load_one(
        kind="unsupported",
        milestone="M7",
        reason="This one is hard to implement and it is scheduled rather than refused, "
        "which is what the unsupported kind means and why the milestone field is required.",
    )
    assert entries[0].kind == "unsupported"


def test_unsupported_without_a_milestone_is_refused():
    with pytest.raises(DivergenceError, match="a schedule with no date on it is a wish"):
        load_one(kind="unsupported")


def test_upstream_without_a_citation_is_refused():
    with pytest.raises(DivergenceError, match="does not get to be said without a citation"):
        load_one(kind="upstream")


def test_pending_without_an_expiry_is_refused():
    with pytest.raises(DivergenceError, match="most load bearing word in software"):
        load_one(kind="pending")


def test_differs_without_saying_how_is_refused():
    with pytest.raises(DivergenceError, match="does not say how"):
        load_one(expect="differs")


def test_a_duplicate_id_is_refused():
    with pytest.raises(DivergenceError, match="two entries cannot make one claim"):
        parse({"divergence": [dict(GOOD), dict(GOOD)]}, registry())


def test_a_missing_spec_link_is_refused():
    entry = dict(GOOD)
    del entry["spec"]
    with pytest.raises(DivergenceError, match="not a decision"):
        parse({"divergence": [entry]}, registry())


# ---------------------------------------------------------------------------
# Expiry, the only calendar in this repository
# ---------------------------------------------------------------------------


def test_an_expired_entry_is_expired():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    entry = load_one(kind="pending", expires=yesterday)[0]
    assert entry.is_expired()


def test_an_unexpired_entry_is_not():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    entry = load_one(kind="pending", expires=tomorrow)[0]
    assert not entry.is_expired()


def test_an_entry_with_no_expiry_never_expires():
    assert not load_one()[0].is_expired()


# ---------------------------------------------------------------------------
# The registry as it ships
# ---------------------------------------------------------------------------


def test_the_committed_registry_loads():
    entries = divergences.registry()
    assert len(entries) == 8
    assert all(isinstance(entry, Divergence) for entry in entries)


def test_every_entry_is_asserted_by_at_least_one_case():
    known = registry()
    for entry in divergences.registry():
        assert any(entry.matches(case_id) for case_id in known), entry.id


def test_nothing_in_the_committed_registry_has_expired():
    """Which is the assertion, not a formality. This test failing is the calendar."""
    assert [entry.id for entry in divergences.registry() if entry.is_expired()] == []


def test_the_registry_file_parses_as_toml():
    tomllib.loads(divergences.REGISTRY.read_text())


def test_the_generated_page_is_current():
    """Generated rather than written, so it cannot fall behind the code."""
    assert divergences.PAGE.exists()
    assert divergences.PAGE.read_text() == divergences.page()


def test_the_page_names_every_entry_and_its_reason():
    text = divergences.page()
    for entry in divergences.registry():
        assert entry.id in text
        assert entry.reason in text


def test_the_inplace_table_covers_every_pandas_callable_that_takes_inplace():
    """The list of names is taken from pandas rather than from memory.

    A hand written list of 42 names is missing three of them within a month, and the
    three it is missing are the ones nobody thought about. This is the check that
    makes the divergence complete rather than representative.
    """
    import json

    from fpcompat import surface
    from fpcompat.cases.divergences import INDEX_INPLACE, INPLACE

    document = json.loads(surface.path_for(pd.__version__).read_text())
    expected = set()
    for space, entry in document["namespaces"].items():
        for member, info in entry["members"].items():
            if any(param["name"] == "inplace" for param in info.get("signature") or []):
                expected.add(f"{space}.{member}")

    covered = {row[0] for row in INPLACE} | {row[0] for row in INDEX_INPLACE}
    covered |= {"MultiIndex.rename", "MultiIndex.set_names", "pandas.eval"}
    assert expected - covered == set(), "pandas callables taking inplace with no case"
    assert covered - expected == set(), "cases claiming an inplace parameter that is gone"
