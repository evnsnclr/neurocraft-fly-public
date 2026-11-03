"""Comparison masks use discovery features, never perturbation outcomes."""

import json

import numpy as np
import pytest

from scripts.circuit_playground_masks import ComparisonMaskError, select_comparison_pairs


def sample_inputs(pairs=40):
    bodies = np.arange(1000, 1002 + 2 * pairs, dtype=np.uint64)
    types = np.asarray(["target", "target", *[f"relay-{i // 2}" for i in range(2 * pairs)]])
    sides = np.asarray(["L", "R"] * (pairs + 1))
    signs = np.asarray([1, -1] * (pairs + 1), dtype=np.int8)
    values = np.asarray([0.65, 0.65, *[0.5 + 0.01 * (i // 2) for i in range(2 * pairs)]])
    features = np.column_stack((values, 10 * values + 1))
    return {
        "body_ids": bodies,
        "types": types,
        "sides": sides,
        "signs": signs,
        "excluded_indices": set(),
        "feature_matrix": features,
        "target_indices": [1, 0],
    }


def test_selection_is_bilateral_same_type_sign_matched_disjoint_and_json_compatible():
    inputs = sample_inputs()
    inputs["excluded_indices"] = {2, 3, 4}
    result = select_comparison_pairs(**inputs)
    assert len(result) == 10
    selected = [index for mask in result for index in mask["indices"]]
    assert len(set(selected)) == 20
    assert not set(selected).intersection({0, 1, 2, 3, 4})
    for mask in result:
        left, right = mask["indices"]
        assert list(inputs["sides"][[left, right]]) == ["L", "R"]
        assert inputs["types"][left] == inputs["types"][right] == mask["type"]
        assert mask["signs"] == [1, -1]
        assert mask["body_ids"] == [int(inputs["body_ids"][i]) for i in [left, right]]
        assert mask["target_indices"] == [0, 1]
        assert mask["match_limits"]["activity_calipers_enforced"] is False
        assert mask["matching_label"] == "size/side/sign matched, activity/strength ranked"
        assert mask["within_original_activity_calipers"] == all(
            mask["side_within_original_activity_calipers"]
        )
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_seed_controls_membership_inside_nearest_pool_and_is_repeatable():
    inputs = sample_inputs(pairs=60)
    first = select_comparison_pairs(**inputs, count=3, seed=17)
    assert first == select_comparison_pairs(**inputs, count=3, seed=17)
    assert first != select_comparison_pairs(**inputs, count=3, seed=19)
    assert all(mask["nearest_pool_size"] == 12 for mask in first)
    assert all(mask["matching_rank"] <= 12 for mask in first)


def test_nearest_pool_contains_best_pairs_and_ranks_distant_features_outside_pool():
    inputs = sample_inputs()
    inputs["feature_matrix"][-2:] = 100
    result = select_comparison_pairs(**inputs, count=1, seed=11)
    assert result[0]["matching_rank"] <= 4
    assert not set(result[0]["indices"]).intersection({80, 81})
    assert result[0]["pair_rms_feature_distance"] < 0.3


def test_sign_mismatch_and_missing_or_different_types_cannot_form_pairs():
    inputs = sample_inputs(pairs=4)
    inputs["feature_matrix"][:] = 2
    inputs["signs"][2] = -1
    inputs["types"][5] = "mismatched"
    inputs["types"][6] = ""
    result = select_comparison_pairs(**inputs, count=1)
    assert result[0]["indices"] == [8, 9]
    with pytest.raises(
        ComparisonMaskError, match=r"only 1.*no structural restrictions were relaxed"
    ):
        select_comparison_pairs(**inputs, count=2)


def test_rank_pairing_within_type_remains_disjoint_when_types_have_many_cells():
    inputs = sample_inputs(pairs=12)
    inputs["types"][2:] = "shared"
    inputs["feature_matrix"][:] = 2
    result = select_comparison_pairs(**inputs, count=8)
    selected = [index for mask in result for index in mask["indices"]]
    assert len(set(selected)) == 16
    assert all(mask["type"] == "shared" for mask in result)


def test_constant_features_are_finite_and_poor_activity_matches_are_explicitly_flagged():
    inputs = sample_inputs()
    inputs["feature_matrix"][:] = 2
    result = select_comparison_pairs(**inputs)
    assert all(mask["pair_rms_feature_distance"] == 0 for mask in result)
    assert all(mask["within_original_activity_calipers"] for mask in result)
    inputs["feature_matrix"][:2, 0] = 3
    result = select_comparison_pairs(**inputs)
    assert len(result) == 10
    assert all(not mask["within_original_activity_calipers"] for mask in result)
    assert all(mask["side_rms_feature_distances"][0] > 2 for mask in result)
    assert all(not mask["match_limits"]["activity_calipers_enforced"] for mask in result)


def test_feature_distances_do_not_remove_structurally_eligible_pairs():
    inputs = sample_inputs(pairs=4)
    inputs["feature_matrix"][:] = 0
    inputs["feature_matrix"][:2] = 1
    result = select_comparison_pairs(**inputs, count=4)
    assert len(result) == 4
    assert {index for mask in result for index in mask["indices"]} == set(range(2, 10))
    assert all(not mask["within_original_activity_calipers"] for mask in result)


def test_input_arrays_and_exclusion_set_are_not_mutated():
    inputs = sample_inputs()
    copies = {key: value.copy() for key, value in inputs.items() if hasattr(value, "copy")}
    select_comparison_pairs(**inputs)
    for key, original in copies.items():
        if isinstance(original, np.ndarray):
            np.testing.assert_array_equal(inputs[key], original)
        else:
            assert inputs[key] == original


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_counts_are_rejected(count):
    with pytest.raises(ComparisonMaskError, match="count"):
        select_comparison_pairs(**sample_inputs(), count=count)


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ComparisonMaskError, match="seed"):
        select_comparison_pairs(**sample_inputs(), seed=seed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("body_ids", [1, 1], "unique positive"),
        ("body_ids", [0, 1], "unique positive"),
        ("body_ids", [True, 2], "integers"),
        ("types", ["a"], "matching length"),
        ("sides", [["L", "R"]], "one-dimensional"),
        ("signs", [1.0] * 82, "integers"),
        ("signs", [2] * 82, "-1, 0 or"),
        ("excluded_indices", {-1}, "outside"),
        ("excluded_indices", {82}, "outside"),
        ("target_indices", [0, 0], "distinct"),
        ("target_indices", [0, 2], "one L and one R"),
        ("target_indices", [0, 3], "share one nonempty"),
        ("feature_matrix", [[1]], "shape"),
        ("feature_matrix", np.empty((82, 0)), "positive features"),
    ],
)
def test_invalid_vectors_indices_and_feature_shapes_are_rejected(field, value, message):
    inputs = sample_inputs()
    inputs[field] = value
    with pytest.raises(ComparisonMaskError, match=message):
        select_comparison_pairs(**inputs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_are_rejected_even_at_excluded_cells(value):
    inputs = sample_inputs()
    inputs["excluded_indices"] = {10}
    inputs["feature_matrix"][10, 0] = value
    with pytest.raises(ComparisonMaskError, match="finite numeric"):
        select_comparison_pairs(**inputs)
