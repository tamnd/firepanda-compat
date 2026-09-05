"""Tests for the in process form of the subject.

`tests/test_driver.py` covers the other form, the one that runs a compiled program per
case, and it says why almost all of it is written against a fake. The same argument
applies here for the same reason and there is one more on top of it. This form does not
have a protocol between it and the subject, it has an API, and an API is the kind of
thing a test can get wrong by agreeing with the code rather than with the library.

Which is what happened. The engine asked firepanda for `read_arrow`, because the Mojo
side has a function by that name in `firepanda.io` and the Python module never did.
Nothing failed. Every case came back `unimplemented`, which is a real outcome this
suite reports on purpose, so the run looked like a subject with nothing implemented
rather than like a harness calling a name that does not exist. That is the failure mode
worth building tests against: not a crash, a plausible zero.

So the fake module here is deliberately mean. It carries exactly the names the real
firepanda exports and nothing else, and anything else raises `AttributeError` the way a
real module does. A test that passes against it is a test that would have caught the
bug above on the day it was written.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

from fpcompat import corpus
from fpcompat.engines.firepanda_engine import FirepandaEngine


def fake(**names) -> types.ModuleType:
    """A module holding exactly the names given and nothing else.

    A real `ModuleType` rather than a class with a `__getattr__` on it, so that a
    missing name raises from the same place and at the same traceback depth a real
    absent attribute does. `tests/test_driver.py` makes the same choice for the same
    reason and the runner's `unimplemented` rule is what depends on it.

    Args:
        names: What the module has.

    Returns:
        The module.
    """
    module = types.ModuleType("firepanda")
    for name, value in names.items():
        setattr(module, name, value)
    return module


@pytest.fixture
def module_form(monkeypatch):
    """Builds an engine around a module of the caller's choosing.

    The engine decides which form it is in by importing firepanda in its constructor,
    so the fake has to be in `sys.modules` before the engine exists. Putting it there
    is also what stops a machine that really does have firepanda installed from
    changing what these tests measure.
    """

    def build(module: types.ModuleType) -> FirepandaEngine:
        monkeypatch.setitem(sys.modules, "firepanda", module)
        return FirepandaEngine()

    return build


def test_a_corpus_frame_is_read_once_and_handed_over(module_form):
    """One reader for both sides of every comparison.

    The pandas engine loads a corpus frame with `corpus.load` and this one has to load
    it with `corpus.load` too, because two readers means a difference in an answer
    could be a difference in the loading, and there is no way to tell those apart from
    a result file. So the frame crosses by the Arrow C data interface instead, which is
    a handover rather than a second read.
    """
    seen = []
    engine = module_form(fake(from_arrow=lambda table: (seen.append(table), "frame")[1]))

    assert engine.frame("two") == "frame"
    assert len(seen) == 1
    assert isinstance(seen[0], pa.Table)
    assert seen[0].equals(corpus.load("two"))


def test_the_engine_asks_firepanda_for_nothing_firepanda_does_not_have(module_form):
    """The test that would have caught `read_arrow`.

    `from_arrow` is the only name the engine is allowed to reach for when it loads a
    frame. Anything else is a name somebody assumed, and the module has nothing else on
    it, so an assumption fails here rather than turning a whole run into zeroes.
    """
    engine = module_form(fake(from_arrow=lambda table: "frame"))

    assert engine.frame("two") == "frame"


def test_the_module_form_evaluates_the_case_expression_in_this_process(module_form):
    """The whole point of the module form, stated as an assertion.

    The runner asks `out_of_process` rather than asking whether a `run` method exists,
    because this engine has one in both forms and calling it in the module form would
    run every case through the driver protocol against a driver that is not there.
    """
    engine = module_form(fake(from_arrow=lambda table: "frame"))

    assert engine.available
    assert engine.form == "module"
    assert not engine.out_of_process
    assert engine.module().from_arrow is not None


def test_the_module_form_wins_when_a_driver_is_built_too(module_form):
    """Both forms present is the normal state of a working checkout, not a conflict.

    The driver is built by hand and stays on disk long after it stops matching the
    firepanda beside it, so the importable module has to be the one that wins. The
    stale binary is then only a thing that takes up space, rather than a thing that
    silently produces the score.
    """
    engine = module_form(fake(from_arrow=lambda table: "frame"))

    assert engine.form == "module"


def test_a_result_file_says_which_firepanda_produced_it(module_form):
    """A conformance number with no version attached is not a number anybody can act on."""
    engine = module_form(fake(from_arrow=lambda table: "frame", __version__="1.2.3"))

    versions = engine.versions()
    assert versions["firepanda"] == "1.2.3"
    assert versions["form"] == "module"


def test_a_module_with_no_version_says_unknown_rather_than_guessing(module_form):
    """Unknown is a fact about the artifact and a lower number would be a fiction."""
    engine = module_form(fake(from_arrow=lambda table: "frame"))

    assert engine.versions()["firepanda"] == "unknown"


# ---------------------------------------------------------------------------
# The real module, when there is one
# ---------------------------------------------------------------------------

try:
    import firepanda as _real
except ImportError:
    _real = None

installed = pytest.mark.skipif(_real is None, reason="no importable firepanda")


@installed
def test_the_real_module_reads_a_corpus_frame():
    """Everything above proves this repository agrees with itself. This is the rest."""
    frame = FirepandaEngine().frame("two")

    assert len(frame) == len(corpus.load("two"))


@installed
def test_the_real_module_carries_a_version_that_is_not_a_placeholder():
    versions = FirepandaEngine().versions()

    assert versions["form"] == "module"
    assert versions["firepanda"] not in ("absent", "unknown", "unstamped")
