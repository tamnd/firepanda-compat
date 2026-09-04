"""Tests for the case registry.

The registry is the part of this repository that decides what a published number
means, so most of these tests are about what it refuses. A case naming a method that
does not exist, a parameter that does not exist, a frame that does not exist or an
exception type that does not exist all have to be fatal at import, because every one
of them produces a coverage number that is wrong in the flattering direction and none
of them is a fact about the library under test.

The registration tests use a fixture that snapshots the module level registry and puts
it back afterwards. Without it a test that registers one case would change the number
the scoreboard prints, which is exactly the sort of test that is worse than no test.
"""

from __future__ import annotations

import pytest

from fpcompat import cases
from fpcompat.cases import ID_PATTERN, LEVELS, OPERATORS, SECTIONS, Case, CaseError, case, registry


@pytest.fixture
def isolated():
    """Registers into a scratch registry and puts the real one back afterwards."""
    saved = dict(cases._REGISTERED)
    saved_section = list(cases._CURRENT_SECTION)
    cases._REGISTERED.clear()
    cases._CURRENT_SECTION.clear()
    cases._CURRENT_SECTION.append("basics")
    yield
    cases._REGISTERED.clear()
    cases._REGISTERED.update(saved)
    cases._CURRENT_SECTION.clear()
    cases._CURRENT_SECTION.extend(saved_section)


def declare(**overrides):
    """One valid case, with the fields a test wants to break passed in."""
    fields = {
        "id": "basics/scratch",
        "api": "DataFrame.head",
        "frames": ("two",),
        "expr": lambda pd, df: df.head(1),
    }
    fields.update(overrides)
    return case(**fields)


# ---------------------------------------------------------------------------
# What the registry refuses
# ---------------------------------------------------------------------------


def test_unknown_api_is_fatal(isolated):
    """This is the test the whole file exists for.

    A misspelled method name has to stop the run rather than produce a case that will
    fail on every engine and look like a conformance result on both.
    """
    with pytest.raises(CaseError, match="does not exist in pandas"):
        declare(api="Series.zfil")


def test_unknown_namespace_is_fatal(isolated):
    with pytest.raises(CaseError, match="which the inventory does not have"):
        declare(api="Frame.head")


def test_unknown_parameter_is_fatal(isolated):
    """The check that keeps the L3 coverage number honest."""
    with pytest.raises(CaseError, match="has no parameter called min_period"):
        declare(api="Series.rolling", level="L3", covers=("min_period",))


def test_known_parameter_is_accepted(isolated):
    declared = declare(api="Series.rolling", level="L3", covers=("min_periods",))
    assert declared.covers == ("min_periods",)


def test_unknown_frame_is_fatal(isolated):
    with pytest.raises(CaseError, match="which the corpus does not have"):
        declare(frames=("not_a_frame",))


def test_no_frames_is_fatal(isolated):
    with pytest.raises(CaseError, match="runs on nothing"):
        declare(frames=())


def test_no_expression_is_fatal(isolated):
    with pytest.raises(CaseError, match="nothing to run"):
        declare(expr=None)


def test_bad_id_is_fatal(isolated):
    with pytest.raises(CaseError, match="not a usable case id"):
        declare(id="Basics/Scratch")


def test_duplicate_id_is_fatal(isolated):
    """Ids are stable forever, so two cases cannot share one."""
    declare()
    with pytest.raises(CaseError, match="already registered"):
        declare()


def test_unknown_level_is_fatal(isolated):
    with pytest.raises(CaseError, match="is not one of"):
        declare(level="L9")


def test_unknown_section_is_fatal():
    with pytest.raises(CaseError, match="is not a parity section"):
        cases.section("not_a_section")


def test_raises_without_l4_is_fatal(isolated):
    with pytest.raises(CaseError, match="makes it an L4 case"):
        declare(level="L2", raises=("KeyError", "nope"))


def test_unknown_exception_type_is_fatal(isolated):
    """Measured, not guessed.

    pandas 3.0.3 has no `NonExistentTimeError`, it raises a plain `ValueError` for a
    wall clock time in the spring gap, and a case that says otherwise used to get all
    the way to the middle of a run before finding out.
    """
    with pytest.raises(CaseError, match="NonExistentTimeError"):
        declare(level="L4", raises=("NonExistentTimeError", "gap"))


def test_known_exception_types_resolve(isolated):
    declared = declare(level="L4", raises=("MergeError", "No common columns"))
    assert declared.raises == ("MergeError", "No common columns")


# ---------------------------------------------------------------------------
# The operator escape hatch
# ---------------------------------------------------------------------------


def test_operator_registers_without_an_inventory_entry(isolated):
    """`df["a"]` is the most written line in pandas and it has no public name."""
    declared = declare(api="DataFrame.__getitem__")
    assert declared.api in OPERATORS


def test_operator_cannot_claim_coverage(isolated):
    """There is no signature to check the claim against, so the claim is refused."""
    with pytest.raises(CaseError, match="nothing to check the claim against"):
        declare(api="DataFrame.__getitem__", level="L3", covers=("key",))


def test_operator_list_is_closed(isolated):
    """A private name that is not on the hand written list is still refused."""
    with pytest.raises(CaseError, match="does not exist in pandas"):
        declare(api="DataFrame.__finalize__")


# ---------------------------------------------------------------------------
# The registry as it actually ships
# ---------------------------------------------------------------------------


def test_every_section_has_cases():
    """A section listed in SECTIONS with nothing in it is a silent hole."""
    declared = registry()
    sections = {item.section for item in declared.values()}
    assert sections == set(SECTIONS)


def test_every_case_is_well_formed():
    for item in registry().values():
        assert isinstance(item, Case)
        assert ID_PATTERN.match(item.id)
        assert item.level in LEVELS
        assert item.frames
        assert item.section in SECTIONS


def test_ids_start_with_their_section():
    """So that a filter on a section name and a filter on an id prefix agree."""
    for item in registry().values():
        assert item.id.startswith(f"{item.section}/"), item.id


def test_relaxed_cases_say_why():
    """A relaxation without a reason is a rule bent for no recorded cause.

    The rule the Rules docstring states is that a reason is required for a relaxation
    or for a tolerance class looser than the default, which is SINGLE. EXACT and
    SINGLE need no reason because neither of them lets anything through.
    """
    loose = {"ACCUMULATION", "STATISTICAL"}
    for item in registry().values():
        if item.rules.relaxations or item.rules.tolerance.name in loose:
            assert item.rules.reason, item.id


def test_l3_cases_cover_something():
    """L3 means a named parameter was exercised, so it has to name one."""
    for item in registry().values():
        if item.level == "L3":
            assert item.covers, item.id


def test_l4_cases_declare_an_exception():
    for item in registry().values():
        if item.level == "L4":
            assert item.raises, item.id
            assert item.raises[1], f"{item.id} accepts any message, which is not a check"


def test_describe_is_serializable():
    import json

    for item in registry().values():
        json.dumps(item.describe())


def test_select_by_section_and_by_id():
    assert cases.select("errors/") and all(
        item.id.startswith("errors/") for item in cases.select("errors/")
    )
    assert cases.select("nested") == [
        item for item in registry().values() if item.section == "nested"
    ]
    assert len(cases.select(None)) == len(registry())
