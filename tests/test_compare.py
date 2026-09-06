"""Tests for the comparison layer.

The most important test in this file is the last one, which compares every frame in
the corpus against itself under zero relaxations and expects exact equality. That is
the smallest possible version of the oracle self test from document 05, and it is
what catches a normalizer that quietly drops a column, mangles an index, or decides
that a null and a NaN are the same thing. Without it, the first bug in the normalizer
gets published as ten firepanda failures, and every one of those is an hour of
somebody's day spent debugging the wrong repository.

The rest of the file is a test per rule. Each one is written as two answers that
differ in exactly one way, because a test that changes two things at once passes for
the wrong reason about half the time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from fpcompat import corpus
from fpcompat.compare import (
    RELAXATIONS,
    VALUE,
    Rules,
    Tolerance,
    _arrow_sort_key,
    _sort_key,
    canonical_type,
    check_error,
    check_warnings,
    compare,
    normalize,
    same,
)

RELAXED = Rules(relaxations=frozenset({"grouped_order"}), reason="a test")


# ---------------------------------------------------------------------------
# The rules a case declares
# ---------------------------------------------------------------------------


def test_a_relaxation_that_does_not_exist_is_refused():
    """The mechanism that keeps the set of relaxations closed."""
    with pytest.raises(ValueError, match="no such relaxation"):
        Rules(relaxations=frozenset({"close_enough"}), reason="it nearly passes")


def test_a_relaxation_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="has to say why"):
        Rules(relaxations=frozenset({"row_order"}))


def test_a_looser_tolerance_class_needs_a_reason():
    with pytest.raises(ValueError, match="has to say why"):
        Rules(tolerance=Tolerance.STATISTICAL)
    Rules(tolerance=Tolerance.STATISTICAL, reason="Welford against two pass")


def test_the_strict_default_needs_no_reason():
    assert Rules().tolerance is Tolerance.SINGLE
    assert Rules().relaxations == frozenset()


def test_every_relaxation_is_written_down():
    """A name with no sentence beside it is a relaxation nobody has to justify."""
    for name, why in RELAXATIONS.items():
        assert len(why) > 80, name


def test_without_drops_one_relaxation():
    """What the "is this relaxation needed" check runs."""
    rules = Rules(relaxations=frozenset({"grouped_order", "row_order"}), reason="a test")
    assert rules.without("row_order").relaxations == frozenset({"grouped_order"})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_the_three_string_widths_are_one_dtype():
    """Arrow's string widths are a storage decision pandas has no dtype for."""
    widths = {canonical_type(t) for t in (pa.string(), pa.large_string(), pa.string_view())}
    assert widths == {"string"}


def test_integer_and_float_widths_are_not_folded():
    assert canonical_type(pa.int32()) != canonical_type(pa.int64())
    assert canonical_type(pa.float32()) != canonical_type(pa.float64())


def test_dictionary_index_width_is_folded_and_ordered_is_not():
    """Two libraries can assign codes differently and both be right, so the codes are
    compared through the categories. The `ordered` flag is a promise and is not."""
    narrow = pa.dictionary(pa.int8(), pa.large_string(), ordered=True)
    wide = pa.dictionary(pa.int32(), pa.string(), ordered=True)
    assert canonical_type(narrow) == canonical_type(wide)
    unordered = pa.dictionary(pa.int8(), pa.large_string(), ordered=False)
    assert canonical_type(narrow) != canonical_type(unordered)


def test_struct_field_order_is_part_of_the_type():
    one = pa.struct([("a", pa.int64()), ("b", pa.int64())])
    other = pa.struct([("b", pa.int64()), ("a", pa.int64())])
    assert canonical_type(one) != canonical_type(other)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_integers_get_no_tolerance_at_any_class():
    left = pd.DataFrame({"a": [1, 2, 3]})
    right = pd.DataFrame({"a": [1, 2, 4]})
    for tolerance in Tolerance:
        rules = Rules(tolerance=tolerance, reason="a test")
        assert not compare(left, right, rules)


def test_the_float_classes_are_the_widths_they_say():
    base = pd.DataFrame({"a": [1.0]})
    for tolerance, inside, outside in (
        (Tolerance.SINGLE, 1e-13, 1e-11),
        (Tolerance.ACCUMULATION, 1e-10, 1e-8),
        (Tolerance.STATISTICAL, 1e-8, 1e-6),
    ):
        rules = Rules(tolerance=tolerance, reason="a test")
        assert compare(base, pd.DataFrame({"a": [1.0 + inside]}), rules), tolerance
        assert not compare(base, pd.DataFrame({"a": [1.0 + outside]}), rules), tolerance


def test_exact_is_a_class_and_not_a_small_number():
    """One unit in the last place apart, which is as close as two float64 values get
    without being the same value. `1.0 + 1e-16` is not, it is `1.0`."""
    left = pd.DataFrame({"a": [1.0]})
    right = pd.DataFrame({"a": [np.nextafter(1.0, 2.0)]})
    assert compare(left, right)
    assert not compare(left, right, Rules(tolerance=Tolerance.EXACT))


def test_nan_equals_nan():
    """Which `==` does not do, and a column of NaN is a perfectly good answer."""
    nan = pd.DataFrame({"a": [float("nan"), 1.0]})
    assert compare(nan, nan.copy())


def test_the_infinities_compare_exactly():
    inf = pd.DataFrame({"a": [float("inf")]})
    assert compare(inf, inf.copy())
    assert not compare(inf, pd.DataFrame({"a": [float("-inf")]}))
    assert not compare(inf, pd.DataFrame({"a": [1e308]}))


def test_infinity_is_never_within_a_tolerance_of_a_finite_number():
    rules = Rules(tolerance=Tolerance.STATISTICAL, reason="a test")
    assert not compare(pd.DataFrame({"a": [float("inf")]}), pd.DataFrame({"a": [1e308]}), rules)


def test_negative_zero_is_a_value_difference_only_where_the_case_says_so():
    """Four cases in the suite are about sign preservation. Everywhere else IEEE is
    right that these are the same number."""
    zero = pd.DataFrame({"a": [0.0, 1.0]})
    negative = pd.DataFrame({"a": [-0.0, 1.0]})
    assert compare(zero, negative)
    assert not compare(zero, negative, Rules(signed_zero=True))


def test_signed_zero_is_seen_through_a_list_column():
    """The fast path is Arrow's own `equals`, which says these are equal because IEEE
    says so, and it has to be turned off rather than trusted for these four cases."""
    left = pa.table({"v": pa.array([[0.0]])})
    right = pa.table({"v": pa.array([[-0.0]])})
    assert compare(left, right)
    assert not compare(left, right, Rules(signed_zero=True))


def test_a_null_is_not_a_nan():
    """The distinction a library conflates first and the one that corrupts data."""
    nan = pa.table({"v": pa.array([float("nan")], type=pa.float64())})
    null = pa.table({"v": pa.array([None], type=pa.float64())})
    verdict = compare(nan, null)
    assert not verdict
    assert "null" in verdict.differences[0]


def test_pandas_turns_a_nan_into_a_null_in_an_arrow_backed_column():
    """Not a rule of this module and worth pinning anyway, because it explains why the
    test above is written against Arrow rather than against pandas. Constructing a
    `float64[pyarrow]` Series from a NaN produces a null, so a test written the
    obvious way would be comparing a null to a null and passing for no reason."""
    assert pa.array(pd.Series([float("nan")], dtype="float64[pyarrow]")).null_count == 1


def test_strings_are_byte_exact_and_neither_side_normalizes():
    """The corpus carries a combining sequence and its precomposed equivalent next to
    each other for exactly this. A library that silently normalizes unicode has
    changed the user's data."""
    values = corpus.load("strings_unicode").column("value").to_pylist()
    combining, precomposed = values[0], values[1]
    assert combining != precomposed
    left = pa.table({"v": pa.array([combining])})
    right = pa.table({"v": pa.array([precomposed])})
    assert not compare(left, right)


def test_an_empty_list_is_not_a_null_list():
    left = pa.table({"v": pa.array([[], None], type=pa.list_(pa.int64()))})
    right = pa.table({"v": pa.array([None, None], type=pa.list_(pa.int64()))})
    assert not compare(left, right)


def test_nested_floats_use_the_tolerance():
    left = pa.table({"v": pa.array([[1.0, 2.0]])})
    right = pa.table({"v": pa.array([[1.0, 2.0 + 1e-13]])})
    assert compare(left, right)
    assert not compare(left, right, Rules(tolerance=Tolerance.EXACT))


# ---------------------------------------------------------------------------
# Categoricals
# ---------------------------------------------------------------------------


def test_an_unused_category_that_did_not_survive_is_a_difference():
    """Which is the whole of what `observed=` is about."""
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b", "z"]))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    verdict = compare(left, right)
    assert not verdict
    assert "z" in verdict.differences[0]


def test_unordered_categories_compare_as_a_set():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"]))
    assert compare(left, right)


def test_ordered_categories_compare_as_a_list():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True))
    assert not compare(left, right)


def test_the_ordered_flag_itself_is_compared():
    left = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True))
    right = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"], ordered=False))
    assert not compare(left, right)


def test_codes_are_compared_through_the_categories():
    """Two libraries can assign codes in a different order and both be right, so long
    as the values they decode to are the same."""
    left = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
    right = pd.Series(pd.Categorical(["a", "b", "a"], categories=["b", "a"]))
    assert list(left.cat.codes) != list(right.cat.codes)
    assert compare(left, right)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def test_the_resolution_is_compared_exactly():
    """A microsecond answer where pandas gives nanoseconds is a failure and not a
    rounding question, because pandas 3.0 carries the resolution on the dtype."""
    stamps = pd.to_datetime(["2024-01-01"])
    left = pd.Series(stamps.astype("datetime64[us]"))
    right = pd.Series(stamps.astype("datetime64[ns]"))
    assert not compare(left, right)


def test_a_timezone_is_compared_by_name_and_not_by_offset():
    """The same instant in `America/New_York` and in `UTC` is not the same answer,
    which matches what pandas does."""
    left = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    right = pd.Series(left.dt.tz_convert("America/New_York"))
    assert not compare(left, right)


def test_a_naive_column_is_not_a_zoned_one():
    left = pd.Series(pd.date_range("2024-01-01", periods=2))
    right = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    assert not compare(left, right)


# ---------------------------------------------------------------------------
# Shape, names and the index
# ---------------------------------------------------------------------------


def test_a_series_is_not_a_one_column_frame():
    frame = pd.DataFrame({"a": [1, 2]})
    assert not compare(frame["a"], frame)


def test_a_column_label_that_is_a_string_is_not_one_that_is_an_integer():
    """`repr(1)` is the string `"1"`, so rendering labels with `repr` alone folds
    these two together. Both frames print identically and they are not the same
    answer."""
    assert not compare(pd.DataFrame({1: [1]}), pd.DataFrame({"1": [1]}))
    assert not compare(pd.Series([1], name=1), pd.Series([1], name="1"))


def test_column_order_is_a_difference():
    left = pd.DataFrame({"a": [1], "b": [2]})
    assert not compare(left, left[["b", "a"]])


def test_a_default_range_index_is_dropped_on_both_sides():
    """The one relaxation applied globally, and the reason a frame carrying a plain 0
    to n-1 index is comparable to one that has no index at all."""
    left = pd.DataFrame({"a": [1, 2, 3]})
    assert compare(left, pa.table({"a": [1, 2, 3]}))


def test_an_index_that_carries_information_is_compared():
    left = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    right = pd.DataFrame({"a": [1, 2]}, index=["x", "z"])
    assert not compare(left, right)


def test_a_named_range_index_is_not_a_default_one():
    left = pd.DataFrame({"a": [1, 2]})
    right = left.rename_axis("row")
    assert not compare(left, right)


def test_strict_index_compares_the_range_index_a_case_is_about():
    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"a": [1, 2]}, index=pd.RangeIndex(2))
    assert compare(left, right, Rules(strict_index=True))
    shifted = pd.DataFrame({"a": [1, 2]}, index=[5, 6])
    assert not compare(left, shifted, Rules(strict_index=True))


def test_a_multiindex_becomes_one_column_per_level():
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]))
    answer = normalize(frame)
    assert answer.n_index == 2
    assert compare(frame, frame.copy())


def test_the_index_level_names_are_compared():
    frame = pd.DataFrame({"v": [1]}, index=pd.Index(["a"], name="k"))
    renamed = frame.rename_axis("key")
    assert not compare(frame, renamed)


def test_an_unnamed_level_is_described_in_words_and_not_as_a_type():
    """The note a reader gets when the level names differ.

    `_label` renders `None` as `NoneType(None)` and has to keep doing so, because the
    comparison needs it to stay apart from the string "None". Showing that to a person
    reads as though pandas had a level named NoneType, so the note spells it out.
    """
    left = pd.DataFrame({"v": [1]}, index=pd.Index(["a"], name="key"))
    right = pd.DataFrame({"v": [1]}, index=pd.Index(["a"]))
    verdict = compare(left, right)
    assert not verdict
    note = "\n".join(verdict.differences)
    assert "index names unnamed, expected 'key'" in note
    assert "NoneType" not in note


def test_the_level_count_is_reported_before_the_level_names():
    """An engine with no index should be told it has no index first.

    This used to report the names first, so a frame with no index at all was told its
    index names were wrong before it was told it had no index, and the names note was
    the one that survived truncation into the summary. The count is the difference
    that matters and it now leads, and because that check returns, it is the only one.
    """
    left = pd.DataFrame({"v": [1, 2]}, index=[5, 6])
    verdict = compare(left, pa.table({"v": [1, 2]}))
    assert not verdict
    assert len(verdict.differences) == 1
    assert "0 index levels, expected 1" in verdict.differences[0]


def test_a_series_name_is_compared():
    assert not compare(pd.Series([1], name="a"), pd.Series([1], name="b"))


def test_a_row_count_difference_stops_before_the_values():
    verdict = compare(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]}))
    assert not verdict
    assert "rows" in verdict.differences[0]
    assert len(verdict.differences) == 1


# ---------------------------------------------------------------------------
# The order relaxations
# ---------------------------------------------------------------------------


def test_row_order_is_a_difference_by_default():
    """Not "sorted before comparing to be safe". An engine that returns the right rows
    in the wrong order has a bug a user will hit."""
    left = pd.DataFrame({"a": [1, 2, 3]})
    assert not compare(left, left.iloc[::-1].reset_index(drop=True))


def test_grouped_order_sorts_by_the_key_columns():
    frame = pd.DataFrame({"k": ["b", "a", "c"], "v": [1, 2, 3]})
    grouped = frame.groupby("k").sum()
    shuffled = grouped.iloc[[2, 0, 1]]
    assert not compare(grouped, shuffled)
    assert compare(grouped, shuffled, RELAXED)


def test_a_relaxation_that_changed_nothing_is_reported_as_unused():
    """So a relaxation declared and never needed is visible without running the case a
    second time with it turned off."""
    grouped = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}).groupby("k").sum()
    verdict = compare(grouped, grouped.copy(), RELAXED)
    assert verdict.equal
    assert verdict.relaxations_used == frozenset()


def test_a_relaxation_that_was_used_is_reported_as_used():
    grouped = pd.DataFrame({"k": ["b", "a"], "v": [1, 2]}).groupby("k").sum()
    verdict = compare(grouped, grouped.iloc[::-1], RELAXED)
    assert verdict.equal
    assert verdict.relaxations_used == frozenset({"grouped_order"})


def test_grouped_order_does_not_hide_a_wrong_value():
    grouped = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]}).groupby("k").sum()
    wrong = grouped.copy()
    wrong.iloc[0, 0] = 99
    assert not compare(grouped, wrong, RELAXED)


def test_an_order_relaxation_works_on_a_string_view_column():
    # Arrow's `take` has no kernel for the view layouts, so reordering a table with a
    # string view key raised out of pyarrow and the runner recorded "the comparison
    # itself raised" in the result file. firepanda's strings are views, so this was
    # every grouped comparison on a text key, which is not an exotic case.
    left = pa.table({"k": pa.array(["b", "a", "c"]), "v": pa.array([1, 2, 3])})
    schema = pa.schema([pa.field("k", pa.string_view()), pa.field("v", pa.int64())])
    shuffled = left.take([2, 0, 1]).cast(schema)
    assert compare(left, shuffled, Rules(relaxations=frozenset({"row_order"}), reason="a test"))


def test_widening_a_view_does_not_hide_a_wrong_value():
    left = pa.table({"k": pa.array(["a", "b"]), "v": pa.array([1, 2])})
    view = pa.table({"k": pa.array(["a", "b"], pa.string_view()), "v": pa.array([1, 99])})
    assert not compare(left, view, Rules(relaxations=frozenset({"row_order"}), reason="a test"))


def test_row_order_sorts_everything():
    left = pd.DataFrame({"a": [3, 1, 2]})
    rules = Rules(relaxations=frozenset({"row_order"}), reason="pandas documents this as undefined")
    assert compare(left, left.sort_values("a").reset_index(drop=True), rules)


# ---------------------------------------------------------------------------
# The two ways of computing the permutation
# ---------------------------------------------------------------------------
#
# Arrow sorts almost everything and the interpreter sorts what is left. The tests
# below are about the seam between the two, which is the only place this can go
# wrong: two permutations produced by two different rules disagree about equal data.


ORDER = Rules(relaxations=frozenset({"row_order"}), reason="a test")


def test_arrow_refuses_a_dictionary_column_and_says_so_rather_than_raising():
    table = pa.table({"k": pa.array(["b", "a"]).dictionary_encode()})
    assert _arrow_sort_key(table, ["k"]) is None


def test_arrow_refuses_a_nested_column():
    table = pa.table({"k": pa.array([[1, 2], [3]])})
    assert _arrow_sort_key(table, ["k"]) is None


def test_arrow_takes_the_ordinary_columns():
    table = pa.table({"a": [3, 1, 2]})
    assert list(_arrow_sort_key(table, ["a"])) == [1, 2, 0]


def test_the_two_sorts_agree_on_which_rows_pair_up():
    """Not on the order they produce, which they genuinely disagree about, because one
    of them compares 10 against 9 as strings. What has to hold is that a table and a
    permutation of it end up paired the same way under either rule."""
    table = pa.table({"a": [9, 10, 1, None, 2]})
    shuffled = table.take([3, 1, 4, 0, 2])
    arrow = [
        table.take(pa.array(_arrow_sort_key(table, ["a"]))),
        shuffled.take(pa.array(_arrow_sort_key(shuffled, ["a"]))),
    ]
    rendered = [
        table.take(pa.array(_sort_key(table, ["a"]))),
        shuffled.take(pa.array(_sort_key(shuffled, ["a"]))),
    ]
    assert arrow[0].equals(arrow[1])
    assert rendered[0].equals(rendered[1])


def test_a_dictionary_key_column_still_compares_under_the_relaxation():
    """The case the fallback exists for. A categorical group key arrives dictionary
    encoded, Arrow will not sort it, and the comparison has to work anyway."""
    frame = pd.DataFrame({"k": pd.Categorical(["b", "a", "c"]), "v": [1, 2, 3]})
    grouped = frame.groupby("k", observed=True).sum()
    assert compare(grouped, grouped.iloc[[2, 0, 1]], RELAXED)


def test_a_nested_column_still_compares_under_the_relaxation():
    frame = pd.DataFrame({"a": [[1, 2], [3], [4, 5]], "b": [1, 2, 3]})
    assert compare(frame, frame.iloc[::-1].reset_index(drop=True), ORDER)


def test_nulls_in_a_key_column_do_not_change_which_rows_pair_up():
    frame = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    assert compare(frame, frame.iloc[[1, 2, 0]].reset_index(drop=True), ORDER)


def test_one_side_refusing_arrow_puts_both_sides_on_the_rendered_sort(monkeypatch):
    """The seam. If the left table sorted in Arrow and the right one in the
    interpreter, the two permutations would follow different rules about where a null
    goes and whether 9 comes before 10, and a case whose sides are equal would fail."""
    import fpcompat.compare as module

    seen = []

    real_arrow, real_rendered = module._arrow_sort_key, module._sort_key
    monkeypatch.setattr(
        module,
        "_arrow_sort_key",
        lambda t, c: (seen.append("arrow"), None if len(seen) > 1 else real_arrow(t, c))[1],
    )
    monkeypatch.setattr(
        module, "_sort_key", lambda t, c: (seen.append("rendered"), real_rendered(t, c))[1]
    )
    frame = pd.DataFrame({"a": [9, 10, 1]})
    assert compare(frame, frame.iloc[::-1].reset_index(drop=True), ORDER)
    assert seen.count("rendered") == 2


def test_a_ten_thousand_row_answer_sorts_in_arrow_rather_than_in_python():
    """The reason any of this was rewritten. Not a timing assertion, which would be
    flaky, but a check that the fast path is the one a plain answer takes, since the
    fallback is correct and would pass every other test in this file while being two
    orders of magnitude slower."""
    table = pa.table({"a": list(range(10000, 0, -1)), "b": [str(i) for i in range(10000)]})
    assert _arrow_sort_key(table, ["a", "b"]) is not None


# ---------------------------------------------------------------------------
# Scalars, tuples and mappings
# ---------------------------------------------------------------------------


def test_a_scalar_carries_its_type():
    assert same(np.int64(5), 5)
    assert not same(np.int64(5), np.int32(5))
    assert not same(5, 5.0)


def test_the_three_spellings_of_missing_stay_apart():
    """A library that returns `None` where pandas returns `NaT` has changed what the
    user sees, and a suite that folded these together would never say so."""
    assert not same(pd.NaT, None)
    assert not same(pd.NA, None)
    assert not same(pd.NaT, pd.NA)
    assert same(pd.NaT, pd.NaT)


def test_a_pandas_nan_survives_the_conversion_as_a_nan():
    """pyarrow converts a pandas object using pandas' own idea of missing, and for a
    numpy backed float column that turns every NaN into an Arrow null. This suite
    insists elsewhere that a null is not a NaN, so letting the conversion fold them
    together on the oracle side would hand a pass to a subject engine that answered
    with a null where pandas answered with a NaN."""
    answer = normalize(pd.Series([1.0, float("nan"), 3.0]))
    assert answer.table.column(VALUE).null_count == 0
    assert np.isnan(answer.table.column(VALUE)[1].as_py())


def test_a_pandas_na_is_still_a_null_after_the_conversion():
    """The other half, and the reason the rule above is narrowed to numpy floats. An
    extension dtype carries a real mask and its missing is a null under any rule, so
    switching pandas' rules off for it would change nothing and it is left alone.

    `Float64` folds a NaN handed to its constructor into `pd.NA`, so this holds one
    missing value and not two, which is pandas being consistent with itself rather
    than anything this suite decided."""
    answer = normalize(pd.Series([1.0, None], dtype="Float64"))
    column = answer.table.column(VALUE)
    assert column.null_count == 1
    assert column[1].as_py() is None


def test_a_nan_and_a_null_in_a_float_column_do_not_compare_equal():
    """The two tests above are only worth having if the comparison they feed still
    tells the two apart once they arrive."""
    assert not same(pd.Series([1.0, float("nan")]), pd.Series([1.0, None], dtype="Float64"))


def test_a_scalar_float_uses_the_tolerance():
    assert same(1.0, 1.0 + 1e-15)
    assert not compare(1.0, 1.0 + 1e-15, Rules(tolerance=Tolerance.EXACT))


def test_a_tuple_answer_compares_element_wise():
    """Which `align`, `factorize` and `divmod` all return."""
    frame = pd.DataFrame({"a": [1]})
    assert same((frame, 1), (frame.copy(), 1))
    assert not same((frame, 1), (frame.copy(), 2))
    assert not same((frame,), (frame.copy(), 1))


def test_a_mapping_answer_compares_by_key():
    assert same({"a": pd.Series([1])}, {"a": pd.Series([1])})
    assert not same({"a": pd.Series([1])}, {"a": pd.Series([2])})
    assert not same({"a": pd.Series([1])}, {"b": pd.Series([1])})


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


def test_an_error_matches_on_the_type_and_a_substring():
    raised = KeyError("column 'z' is not in the frame")
    assert check_error(raised, "KeyError", "z")
    assert not check_error(raised, "ValueError", "z")
    assert not check_error(raised, "KeyError", "not a substring of anything")


def test_a_subclass_is_not_a_match_on_the_pandas_side():
    """A case has to declare what pandas raises as precisely as pandas raises it.

    `MergeError` is a `ValueError`, and a case that declares the base class states
    less than it knows. Accepting it would let the suite record an agreement on a
    type it never compared."""
    assert issubclass(pd.errors.MergeError, ValueError)
    assert not check_error(pd.errors.MergeError("bad merge"), "ValueError", "bad")
    assert check_error(pd.errors.MergeError("bad merge"), "MergeError", "bad")


def test_a_subclass_is_a_match_on_the_subject_side():
    """Because the question there is whether an except clause still fires.

    Code written against pandas catches `KeyError`, and a subclass of `KeyError` is
    caught by it, so an engine raising one has not broken anybody. This is how
    firepanda's `ColumnNotFoundError` passes a case that declares `KeyError`, and it
    is the same arrangement pandas has with its own 46 error types."""

    class ColumnNotFound(KeyError):
        pass

    assert check_error(ColumnNotFound("no column 'z'"), "KeyError", "z", exact=False)
    assert not check_error(ColumnNotFound("no column 'z'"), "KeyError", "z")


def test_a_superclass_is_never_a_match_on_either_side():
    """The half of the exact rule that was doing the real work.

    If pandas raises `MergeError` and the subject raises a plain `ValueError`, every
    `except MergeError` a user wrote stops firing, so relaxing the rule in one
    direction must not relax it in the other."""
    assert not check_error(ValueError("bad merge"), "MergeError", "bad", exact=False)
    assert not check_error(Exception("bad merge"), "ValueError", "bad", exact=False)


def test_the_message_still_has_to_match_for_a_subclass():
    """A subclass gets no discount on the substring, which is the other half."""

    class ColumnNotFound(KeyError):
        pass

    verdict = check_error(ColumnNotFound("something went wrong"), "KeyError", "z", exact=False)
    assert not verdict
    assert "does not contain" in verdict.differences[0]


def test_the_pandas_error_types_resolve_by_their_short_name():
    for name in ("MergeError", "ParserError", "OutOfBoundsDatetime", "IntCastingNaNError"):
        assert check_error(getattr(pd.errors, name)("x"), name, "x")


def test_nothing_raised_where_something_was_expected():
    verdict = check_error(None, "KeyError", "z")
    assert not verdict
    assert "nothing was raised" in verdict.differences[0]


def test_a_case_that_names_a_type_that_does_not_exist_stops_the_run():
    """A broken case is not a conformance failure and must not be counted as one."""
    with pytest.raises(LookupError):
        check_error(KeyError("x"), "NoSuchErrorType", "x")


def test_a_warning_that_should_not_have_fired_is_a_difference():
    """The failure a user finds as a crash under `-W error`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("something", FutureWarning, stacklevel=1)
    assert not check_warnings(list(caught), None)
    assert check_warnings([], None)


def test_a_warning_matches_on_the_type_and_a_substring():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("the frobnicator is deprecated", FutureWarning, stacklevel=1)
    caught = list(caught)
    assert check_warnings(caught, ("FutureWarning", "frobnicator"))
    assert not check_warnings(caught, ("DeprecationWarning", "frobnicator"))
    assert not check_warnings(caught, ("FutureWarning", "widget"))


# ---------------------------------------------------------------------------
# An engine that is not pandas
# ---------------------------------------------------------------------------
#
# All of this is written against a fake built out of pyarrow rather than against
# firepanda, and not only so the tests run on a machine with no firepanda on it. The
# bug being fixed here was this module knowing only pandas, pyarrow and numpy, and a
# test that reached for firepanda to prove the fix would be re-teaching it one more
# name. The fake carries the Arrow dunders and nothing else, which is the whole of
# what any subject is required to have.


class OtherFrame:
    """A frame belonging to some engine that is not pandas."""

    def __init__(self, table: pa.Table, index: object = None) -> None:
        self._table = table
        self.index = index

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self._table.__arrow_c_stream__(requested_schema)


class OtherColumn:
    """One column belonging to that engine, which is a series or an index or neither."""

    def __init__(self, array: pa.Array, index: object = None, name: object = None) -> None:
        self._array = array
        self.index = index
        self.name = name

    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, ...]:
        return self._array.__arrow_c_array__(requested_schema)


def other_frame(values: dict, labels: list | None = None, name: object = None) -> OtherFrame:
    """Builds a foreign frame, with row labels when it is given some.

    Args:
        values: The data columns.
        labels: The row labels, or None for a frame that does not have any.
        name: The index name.

    Returns:
        The frame.
    """
    index = None if labels is None else OtherColumn(pa.array(labels), name=name)
    return OtherFrame(pa.table(values), index=index)


def test_a_frame_from_another_engine_is_a_frame_and_not_a_scalar():
    """The regression. 41 of the 167 failures in the first run against the importable
    firepanda were this, and not one of them was a bug in firepanda."""
    answer = normalize(other_frame({"a": [1, 2, 3]}))

    assert answer.kind == "frame"
    assert answer.columns == ("a",)


def test_a_frame_from_another_engine_compares_equal_to_the_pandas_answer():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    assert compare(frame, other_frame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))


def test_a_frame_from_another_engine_carries_its_row_labels():
    """A frame's Arrow export is its data columns and its labels are a separate object,
    so a conversion that read only the stream would agree with pandas about a `tail`
    that had thrown its labels away."""
    frame = pd.DataFrame({"a": [1, 2, 3]}, index=[5, 6, 7])

    assert compare(frame, other_frame({"a": [1, 2, 3]}, labels=[5, 6, 7]))


def test_a_frame_whose_labels_are_wrong_fails():
    """The test that has to stay. A correct frame passes under the broken version too,
    once it is a frame at all, so the assertion that goes quiet if somebody simplifies
    the conversion down to the data columns is this one and not the one above."""
    frame = pd.DataFrame({"a": [1, 2, 3]}, index=[5, 6, 7])

    assert not compare(frame, other_frame({"a": [1, 2, 3]}, labels=[0, 1, 2]))


def test_a_frame_with_no_labels_at_all_fails_against_a_pandas_answer_that_has_some():
    """A library with no index compared against `tail` is the shape of this, and the
    message it produces is the one every such failure has to produce."""
    frame = pd.DataFrame({"a": [1, 2, 3]}, index=[5, 6, 7])
    verdict = compare(frame, other_frame({"a": [1, 2, 3]}))

    assert not verdict
    assert "index levels" in verdict.summary()


def test_a_default_range_of_labels_is_still_dropped_on_both_sides():
    """The pandas test for this asks whether the index is a `RangeIndex`, which cannot
    be asked of another library, so the labels are read instead."""
    frame = pd.DataFrame({"a": [1, 2, 3]})

    assert compare(frame, other_frame({"a": [1, 2, 3]}, labels=[0, 1, 2]))
    assert compare(frame, other_frame({"a": [1, 2, 3]}))


def test_a_named_index_of_zero_to_n_is_not_a_default_index():
    frame = pd.DataFrame({"a": [1, 2, 3]})

    assert not compare(frame, other_frame({"a": [1, 2, 3]}, labels=[0, 1, 2], name="k"))


def test_a_column_from_another_engine_is_a_series_when_the_engine_says_so():
    series = pd.Series([1, 2, 3], name="a")
    column = OtherColumn(pa.array([1, 2, 3]), index=OtherColumn(pa.array([0, 1, 2])), name="a")

    assert normalize(column, "series").kind == "series"
    assert compare(series, normalize(column, "series"))


def test_a_series_from_another_engine_with_the_wrong_name_fails():
    series = pd.Series([1, 2, 3], name="a")
    column = OtherColumn(pa.array([1, 2, 3]), index=OtherColumn(pa.array([0, 1, 2])), name="b")

    assert not compare(series, normalize(column, "series"))


def test_a_series_from_another_engine_carries_its_row_labels():
    series = pd.Series([1, 2, 3], index=[5, 6, 7], name="a")
    labels = OtherColumn(pa.array([5, 6, 7]))
    right = OtherColumn(pa.array([1, 2, 3]), index=labels, name="a")
    wrong = OtherColumn(pa.array([1, 2, 3]), index=OtherColumn(pa.array([0, 1, 2])), name="a")

    assert compare(series, normalize(right, "series"))
    assert not compare(series, normalize(wrong, "series"))


def test_an_index_from_another_engine_is_an_index_when_the_engine_says_so():
    """And this is the one distinction the Arrow interface cannot make on its own,
    which is the entire reason `Engine.shape_of` exists."""
    column = OtherColumn(pa.array([5, 6, 7]))

    assert normalize(column, "index").kind == "index"
    assert normalize(column).kind == "array"
    assert compare(pd.Index([5, 6, 7]), normalize(column, "index"))
    assert not compare(pd.Index([5, 6, 7]), normalize(column))


def test_an_index_from_another_engine_with_the_wrong_name_fails():
    column = OtherColumn(pa.array([5, 6, 7]), name="k")

    assert not compare(pd.Index([5, 6, 7]), normalize(column, "index"))


def test_a_pandas_object_does_not_take_the_arrow_route():
    """pandas offers the same dunders, and it has to keep going down the branch that
    knows about its labels, its column names and its extension dtypes."""
    frame = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([5, 6, 7], name="k"))
    answer = normalize(frame)

    assert answer.kind == "frame"
    assert answer.n_index == 1
    assert answer.index_names == ("k",)


def test_an_answer_that_is_not_arrow_at_all_is_still_a_scalar():
    """A subject producing neither pandas objects nor Arrow lands here, and for that
    subject this is the right answer rather than a gap."""
    assert normalize(7).kind == "scalar"
    assert normalize("seven").kind == "scalar"


# ---------------------------------------------------------------------------
# The self test
# ---------------------------------------------------------------------------


def test_every_corpus_frame_equals_itself_exactly():
    """The smallest version of the oracle, and the most valuable test in this file.

    Every frame in the corpus, through the whole normalizer, against a copy of
    itself, with zero relaxations and the exact tolerance class. Anything other than
    a perfect result here is a bug in this module, and it would otherwise be published
    as a firepanda failure and cost somebody a day in the wrong repository.
    """
    rules = Rules(tolerance=Tolerance.EXACT)
    failures = {}
    for name in corpus.frames():
        frame = corpus.load(name).to_pandas()
        verdict = compare(frame, frame.copy(), rules)
        if not verdict:
            failures[name] = verdict.summary()
    assert failures == {}


def test_every_corpus_frame_equals_itself_through_arrow_as_well():
    """The same run without pandas in the middle, so that a bug in `to_pandas` cannot
    make the test above pass by damaging both sides identically."""
    rules = Rules(tolerance=Tolerance.EXACT)
    for name in corpus.frames():
        table = corpus.load(name)
        assert compare(table, corpus.load(name), rules), name


def test_a_single_changed_value_is_caught_in_every_corpus_frame():
    """The other half of the test above. A comparison that says everything is equal is
    only worth something if it also says when something is not, and a normalizer that
    dropped every column would pass the self test perfectly."""
    for name in corpus.frames():
        frame = corpus.load(name).to_pandas()
        if frame.empty:
            continue
        damaged = frame.copy()
        column = damaged.columns[-1]
        # Overwriting with a string changes either the value or the dtype, and either
        # one has to be a difference, which is the point.
        damaged[column] = damaged[column].astype(object)
        damaged.iloc[0, damaged.columns.get_loc(column)] = "damaged"
        assert not compare(frame, damaged), name
