"""Tests for the corpus generator.

Like the surface tests, these are tests of the instrument. A corpus that generates
differently on two machines, or that loses a distinction on the way to disk, does
not produce a wrong conformance number. It produces a conformance number that is
about nothing, which is worse, because nobody looking at the report can tell.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pyarrow as pa
import pytest

from fpcompat import corpus


@pytest.fixture(scope="module")
def built():
    return corpus.frames()


def test_the_generator_produces_the_words_it_is_supposed_to():
    """The first four words of the stream from the shared seed, written down.

    firepanda-bench and `firepanda/testing/rng.mojo` produce this same stream from
    this same seed, and the point of writing the words here is that a change to any
    of the three constants fails a test in this repository instead of quietly making
    two repositories describe different data with the same words.
    """
    assert [int(word) for word in corpus.splitmix64(corpus.SEED, 4)] == [
        3220344897584144929,
        10671001446143789449,
        15948751857155702275,
        15830066176122234880,
    ]
    assert len({int(word) for word in corpus.splitmix64(corpus.SEED, 1000)}) == 1000


def test_the_generator_agrees_with_the_bench_if_it_is_checked_out():
    """The same stream from the other implementation of it.

    Skipped when the bench is not beside this checkout, for the same reason the
    parity gap test is skipped without the library: one checkout should be enough to
    run the tests. The literal words above are what holds when it is skipped.
    """
    import sys

    tools = corpus.ROOT.parent / "firepanda-bench" / "tools"
    if not (tools / "data.py").exists():
        pytest.skip("firepanda-bench checkout not beside this one")
    sys.path.insert(0, str(tools))
    try:
        import data
    finally:
        sys.path.remove(str(tools))
    assert data.GOLDEN == corpus.GOLDEN
    assert data.MIX_A == corpus.MIX_A
    assert data.MIX_B == corpus.MIX_B
    mine = corpus.splitmix64(corpus.SEED, 256, skip=7)
    theirs = data.splitmix64(corpus.SEED, 256, skip=7)
    assert [int(word) for word in mine] == [int(word) for word in theirs]


def test_the_counter_form_slices_the_same_stream():
    """Any window of the stream computes without computing the words before it."""
    whole = corpus.splitmix64(corpus.SEED, 100)
    tail = corpus.splitmix64(corpus.SEED, 40, skip=60)
    assert [int(word) for word in whole[60:]] == [int(word) for word in tail]


def test_generation_is_deterministic():
    """Two runs in one process, which is the weakest form of this and still catches
    anything that reached for `hash`, `id`, the clock or an unseeded generator."""
    first = corpus.manifest()
    second = corpus.manifest()
    assert corpus.dumps(first) == corpus.dumps(second)


def test_nothing_depends_on_the_python_hash_seed(tmp_path):
    """The real form of the determinism test, in a fresh interpreter with a
    different hash seed. `hash` on a string is salted per process, so a corpus that
    used it would pass the test above every time and still differ between two
    developers."""
    import os
    import subprocess
    import sys

    script = "from fpcompat import corpus; print(corpus.dumps(corpus.manifest()), end='')"
    outputs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=corpus.ROOT,
            env=env,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def test_every_frame_round_trips_to_the_same_digest(built):
    """The test that sent the frames from Parquet to Arrow IPC.

    Parquet came back with `timestamp[s]` promoted to `timestamp[ms]`, with the
    `large_string` inside a dictionary narrowed to `string`, and with a large list
    child renamed. A case reads what `load` returns, so anything the file format
    quietly changes is a distinction the whole corpus stops testing.
    """
    corpus.write_frames(built)
    for name, table in built.items():
        loaded = corpus.load(name)
        assert loaded.schema == table.schema, name
        assert corpus.digest(loaded) == corpus.digest(table), name


def test_the_manifest_on_disk_is_current(built):
    """The same check CI runs."""
    if not corpus.MANIFEST.exists():
        pytest.fail("no corpus/manifest.json: run `pixi run corpus` and commit it")
    committed = json.loads(corpus.MANIFEST.read_text())
    assert corpus.compare(corpus.manifest(), committed) == []


def test_the_manifest_describes_every_frame(built):
    doc = corpus.manifest()
    assert set(doc["frames"]) == set(built)
    for name, record in doc["frames"].items():
        assert record["rows"] == built[name].num_rows
        assert record["columns"], name
        assert len(record["digest"]) == 64, name


def test_every_frame_stays_small(built):
    """The rule from document 04, enforced rather than remembered."""
    for name, table in built.items():
        assert table.num_rows <= corpus.TALL, name


def test_the_float_edges_are_where_the_manifest_says(built):
    for name in ("float64_no_nulls", "float32_no_nulls"):
        values = built[name].column("value").to_pylist()
        offsets = corpus.manifest()["frames"][name]["edges"]
        assert math.isnan(values[offsets["nan"]])
        assert values[offsets["positive_infinity"]] == math.inf
        assert values[offsets["negative_infinity"]] == -math.inf
        zero = values[offsets["negative_zero"]]
        assert zero == 0.0 and math.copysign(1.0, zero) == -1.0


def test_float32_narrows_the_two_values_it_cannot_hold(built):
    """Not a bug and worth pinning: the float64 denormal becomes zero in float32 and
    the float64 maximum becomes infinity. That is what pandas does with the same
    cast, so it is the answer the case is comparing against."""
    values = built["float32_no_nulls"].column("value").to_pylist()
    offsets = corpus.manifest()["frames"]["float32_no_nulls"]["edges"]
    assert values[offsets["smallest_denormal"]] == 0.0
    assert values[offsets["largest_finite"]] == math.inf


def test_the_digest_separates_null_from_nan_and_zero_from_negative_zero():
    """The property that makes the digest worth having."""
    null = pa.table({"a": pa.array([None], type=pa.float64())})
    nan = pa.table({"a": pa.array([float("nan")], type=pa.float64())})
    zero = pa.table({"a": pa.array([0.0], type=pa.float64())})
    negative = pa.table({"a": pa.array([-0.0], type=pa.float64())})
    digests = {corpus.digest(table) for table in (null, nan, zero, negative)}
    assert len(digests) == 4


def test_integer_edges_carry_the_limit_of_every_width(built):
    table = built["integer_edges"]
    for name, dtype in corpus.INT_WIDTHS.items():
        values = table.column(name).to_pylist()
        bits = dtype.bit_width
        if pa.types.is_signed_integer(dtype):
            assert values[0] == -(1 << (bits - 1))
            assert values[1] == (1 << (bits - 1)) - 1
        else:
            assert values[0] == 0
            assert values[1] == (1 << bits) - 1


def test_the_null_shapes_are_what_they_claim(built):
    for name in ("int64_no_nulls", "int64_half_null", "int64_all_null"):
        column = built[name].column("value")
        expected = {"no_nulls": 0, "half_null": corpus.ROWS // 2, "all_null": corpus.ROWS}[
            name.removeprefix("int64_")
        ]
        assert column.null_count == expected, name


def test_half_null_is_not_contiguous(built):
    """A validity implementation that only reads the first word of the bitmap passes
    a half null column whose nulls are all at the front."""
    values = built["int64_half_null"].column("value").to_pylist()
    assert values[0] is not None
    assert values[1] is None
    assert values[2] is not None


def test_the_ascii_strings_straddle_the_stringview_boundary(built):
    lengths = [len(value) for value in built["strings_ascii"].column("value").to_pylist()]
    assert lengths == list(range(21))
    assert 12 in lengths and 13 in lengths


def test_null_and_empty_string_are_both_present(built):
    values = built["strings_null_heavy"].column("value").to_pylist()
    assert None in values
    assert "" in values


def test_the_awkward_keys_carry_a_null_and_an_empty_string(built):
    keys = built["keys_awkward"].column("key").to_pylist()
    assert None in keys
    assert "" in keys


def test_the_two_composed_forms_of_cafe_are_different_strings(built):
    values = built["strings_unicode"].column("value").to_pylist()
    assert values[0] != values[1]
    assert len(values[0]) != len(values[1])


def test_the_categories_are_not_already_sorted(built):
    """Sorting an ordered categorical follows the category order, and a corpus whose
    categories happen to be alphabetical cannot tell that from a lexical sort."""
    column = built["categorical_ordered"].column("value").chunk(0)
    categories = column.dictionary.to_pylist()
    assert categories != sorted(categories)
    assert column.type.ordered


def test_the_unused_category_survives_to_pandas(built):
    series = built["categorical_ordered"].to_pandas()["value"]
    assert "unused" in list(series.cat.categories)
    assert "unused" not in set(series.dropna())


def test_the_resolutions_stay_distinct(built):
    types = {field.name: str(field.type) for field in built["temporal_resolutions"].schema}
    assert types["s"] == "timestamp[s]"
    assert types["ms"] == "timestamp[ms]"
    assert types["us"] == "timestamp[us]"
    assert types["ns"] == "timestamp[ns]"


def test_the_dst_frames_cross_the_transition(built):
    """New York moves by an hour and Lord Howe Island moves by thirty minutes, which
    is the one that finds code assuming a whole hour."""
    for name, expected in (
        ("temporal_dst_forward", pd.Timedelta(hours=1)),
        ("temporal_dst_back", pd.Timedelta(hours=-1)),
        ("temporal_dst_lord_howe", pd.Timedelta(minutes=30)),
    ):
        series = built[name].to_pandas()["zoned"]
        offsets = {value.utcoffset() for value in series}
        assert len(offsets) == 2, name
        low, high = sorted(offsets)
        assert high - low == abs(expected), name


def test_the_temporal_range_leaves_the_nanosecond_window(built):
    """Rows outside the range a nanosecond timestamp can hold, which is how one finds
    out whether something still assumes nanoseconds."""
    values = built["temporal_range"].column("second").to_pylist()
    assert min(values).year < 1678
    assert max(values).year > 2262


def test_the_key_frames_have_the_cardinalities_they_are_named_for(built):
    for name, groups in (("keys_10", 10), ("keys_1000", 1000), ("keys_unique", corpus.TALL)):
        assert built[name].column("key").to_pandas().nunique() == groups, name


def test_the_two_column_keys_are_not_two_one_column_keys(built):
    frame = built["keys_two_column"].to_pandas()
    assert frame["left"].nunique() < len(frame)
    assert frame["right"].nunique() < len(frame)
    assert len(frame.groupby(["left", "right"])) > max(
        frame["left"].nunique(), frame["right"].nunique()
    )


def test_pandas_can_read_every_frame(built):
    """The oracle has to be able to hold the whole corpus. A frame pandas cannot
    represent has nothing to compare against and does not belong here, which is why
    Int128 is not in the numeric section."""
    for name, table in built.items():
        frame = table.to_pandas()
        assert len(frame) == table.num_rows, name


def test_the_check_mode_fails_when_the_manifest_disagrees(built):
    doc = corpus.manifest()
    tampered = json.loads(corpus.dumps(doc))
    tampered["frames"]["int64_no_nulls"]["digest"] = "0" * 64
    problems = corpus.compare(doc, tampered)
    assert len(problems) == 1
    assert "int64_no_nulls" in problems[0]


def test_the_check_mode_reports_a_frame_that_appeared_or_vanished(built):
    doc = corpus.manifest()
    tampered = json.loads(corpus.dumps(doc))
    tampered["frames"]["invented"] = tampered["frames"].pop("int64_no_nulls")
    problems = corpus.compare(doc, tampered)
    assert any("in the corpus and not in the manifest" in line for line in problems)
    assert any("in the manifest and not in the corpus" in line for line in problems)
