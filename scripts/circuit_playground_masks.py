"""Discovery-only ranking of structurally matched internal comparison masks.

The feature columns are supplied by the caller (for example unperturbed activity
and log connection strength). They must not contain perturbation outcomes. A
comparison's two neurons share an annotation type with *each other*, not
necessarily with the target: excluding a two-cell DNp01 target otherwise leaves
no DNp01 comparisons. Each side must have the target side's presynaptic sign.

Every feature is normalized using 1.4826 * MAD over eligible cells plus targets,
with only a 64-epsilon numerical floor. Activity and strength rank candidates;
they are not eligibility limits. Original component-3 / side-RMS-2 calipers are
reported only as descriptive match-quality flags. A structurally eligible mask
can be a poor activity/strength match, and its descriptor says so. Insufficient
structural matches raise an error; anatomical restrictions are never relaxed.

Within each type, eligible left and right cells are ranked separately by distance
to their respective target, then paired by rank. This produces disjoint masks
without a quadratic Cartesian product. Among the best 4 * count such pairs, a
seeded sample without replacement supplies the requested masks. This is a
descriptive comparison sample, not a biological null distribution.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAX_FEATURE_DISTANCE = 3.0
MAX_SIDE_RMS_DISTANCE = 2.0
NEAREST_POOL_MULTIPLIER = 4
Vector = Sequence[object] | NDArray[Any]


class ComparisonMaskError(ValueError):
    """Invalid inputs or insufficient structurally eligible comparison masks."""


def _integers(values: Collection[object] | NDArray[Any], label: str) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ComparisonMaskError(f"{label} must contain integers, not booleans")
        result.append(int(value))
    return result


def _vector(values: Vector, label: str, length: int | None = None) -> NDArray[Any]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ComparisonMaskError(f"{label} must be a one-dimensional vector of matching length")
    return array


def select_comparison_pairs(
    body_ids: Vector,
    types: Vector,
    sides: Vector,
    signs: Vector,
    excluded_indices: Collection[int],
    feature_matrix: ArrayLike,
    target_indices: Collection[int],
    count: int = 10,
    seed: int = 20261024,
) -> list[dict[str, Any]]:
    """Return JSON-compatible L/R masks, ranked by discovery-feature distances.

    ``feature_matrix`` has shape (number of bodies, number of discovery features).
    Unknown type/side annotations may be None or strings; only nonempty types
    with exactly L/R candidate sides qualify. ``signs`` contains -1, 0 or +1.
    Target indices must name one left and one right neuron of the same type.
    The function does not load data, mutate inputs, fit thresholds, or inspect
    lesion outcomes. Selection details and the actual match distances accompany
    every mask. Feature distances never filter eligible masks. A structural
    shortage raises ComparisonMaskError before returning any masks.
    """
    if type(count) is not int or count < 1:
        raise ComparisonMaskError("count must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ComparisonMaskError("seed must be an integer in [0, 2**64)")
    bodies = _integers(_vector(body_ids, "body_ids"), "body_ids")
    size = len(bodies)
    if size < 2 or min(bodies) < 1 or len(set(bodies)) != size:
        raise ComparisonMaskError("body_ids must contain unique positive integers")
    type_array = _vector(types, "types", size)
    side_array = _vector(sides, "sides", size)
    for label, values in (("types", type_array), ("sides", side_array)):
        if any(value is not None and not isinstance(value, str) for value in values):
            raise ComparisonMaskError(f"{label} must contain strings or None")
    sign_values = _integers(_vector(signs, "signs", size), "signs")
    if any(value not in {-1, 0, 1} for value in sign_values):
        raise ComparisonMaskError("signs must be -1, 0 or +1")
    targets = _integers(target_indices, "target_indices")
    excluded = _integers(excluded_indices, "excluded_indices")
    if len(targets) != 2 or len(set(targets)) != 2:
        raise ComparisonMaskError("target_indices must contain exactly two distinct indices")
    if any(index < 0 or index >= size for index in (*targets, *excluded)):
        raise ComparisonMaskError("target/excluded index outside the body vector")
    if sorted(side_array[index] for index in targets if side_array[index] in {"L", "R"}) != [
        "L",
        "R",
    ]:
        raise ComparisonMaskError("target pair must contain exactly one L and one R neuron")
    left_target, right_target = sorted(targets, key=lambda index: str(side_array[index]))
    target_type = type_array[left_target]
    if not target_type or target_type != type_array[right_target]:
        raise ComparisonMaskError("target pair must share one nonempty annotation type")
    try:
        features = np.asarray(feature_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ComparisonMaskError("feature_matrix must contain finite numeric features") from exc
    if features.ndim != 2 or features.shape[0] != size or features.shape[1] < 1:
        raise ComparisonMaskError("feature_matrix must have shape (bodies, positive features)")
    if not bool(np.isfinite(features).all()):
        raise ComparisonMaskError("feature_matrix must contain finite numeric features")

    excluded_set = set(excluded) | set(targets)
    eligible = [
        index
        for index in range(size)
        if index not in excluded_set
        and type_array[index]
        and side_array[index] in {"L", "R"}
        and sign_values[index]
        == sign_values[left_target if side_array[index] == "L" else right_target]
    ]
    reference = features[[*eligible, *targets]]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center = np.median(reference, axis=0)
        mad = np.median(np.abs(reference - center), axis=0)
        floor = 64 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(center))
        scale = np.maximum(1.4826 * mad, floor)
    if not bool(np.isfinite(scale).all()) or not bool(np.isfinite(center).all()):
        raise ComparisonMaskError("feature magnitude exceeds finite matching arithmetic")

    ranked_by_type: dict[str, dict[str, list[tuple[float, int, list[float]]]]] = {}
    for index in eligible:
        side = str(side_array[index])
        target = left_target if side == "L" else right_target
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            differences = np.abs((features[index] - features[target]) / scale)
        if not bool(np.isfinite(differences).all()):
            raise ComparisonMaskError("feature magnitude exceeds finite matching arithmetic")
        rms = math.hypot(*(float(value) for value in differences)) / math.sqrt(features.shape[1])
        if not math.isfinite(rms):
            raise ComparisonMaskError("feature magnitude exceeds finite matching arithmetic")
        group = ranked_by_type.setdefault(str(type_array[index]), {"L": [], "R": []})
        group[side].append((rms, index, differences.tolist()))
    pairs = []
    for annotation_type, group in ranked_by_type.items():
        for members in group.values():
            members.sort(key=lambda member: (member[0], bodies[member[1]]))
        for left, right in zip(group["L"], group["R"], strict=False):
            pair_distance = math.hypot(left[0] / math.sqrt(2), right[0] / math.sqrt(2))
            pairs.append((pair_distance, annotation_type, left, right))
    pairs.sort(key=lambda pair: (pair[0], pair[1], bodies[pair[2][1]], bodies[pair[3][1]]))
    if len(pairs) < count:
        raise ComparisonMaskError(
            f"requested {count} comparison pairs; only {len(pairs)} disjoint bilateral pairs "
            "satisfy same-type/side/sign restrictions and exclusions; "
            "no structural restrictions were relaxed"
        )
    pool_size = min(len(pairs), NEAREST_POOL_MULTIPLIER * count)
    chosen = np.random.Generator(np.random.PCG64(seed)).choice(pool_size, size=count, replace=False)
    # Report in match-rank order; RNG choice controls membership, not presentation.
    result = []
    for rank in sorted(int(value) for value in chosen):
        distance, annotation_type, left, right = pairs[rank]
        indices = [left[1], right[1]]
        within_calipers = [
            max(member[2]) <= MAX_FEATURE_DISTANCE and member[0] <= MAX_SIDE_RMS_DISTANCE
            for member in (left, right)
        ]
        result.append(
            {
                "indices": indices,
                "body_ids": [bodies[index] for index in indices],
                "type": annotation_type,
                "sides": ["L", "R"],
                "signs": [sign_values[index] for index in indices],
                "features": [features[index].tolist() for index in indices],
                "target_indices": [left_target, right_target],
                "target_body_ids": [bodies[left_target], bodies[right_target]],
                "target_features": features[[left_target, right_target]].tolist(),
                "standardized_absolute_differences": [left[2], right[2]],
                "side_rms_feature_distances": [left[0], right[0]],
                "pair_rms_feature_distance": distance,
                "matching_rank": rank + 1,
                "eligible_disjoint_pair_count": len(pairs),
                "nearest_pool_size": pool_size,
                "seed": seed,
                "matching_label": "size/side/sign matched, activity/strength ranked",
                "within_original_activity_calipers": all(within_calipers),
                "side_within_original_activity_calipers": within_calipers,
                "match_limits": {
                    "structural": "two cells; one L/R; shared type; target-side signs; exclusions",
                    "original_component_caliper": MAX_FEATURE_DISTANCE,
                    "original_side_rms_caliper": MAX_SIDE_RMS_DISTANCE,
                    "activity_calipers_enforced": False,
                    "activity_strength": "ranking only; calipers are descriptive quality flags",
                },
                "normalization": {
                    "method": "1.4826 * MAD with 64-epsilon numerical floor",
                    "reference": "eligible side/sign cells plus targets",
                    "median": center.tolist(),
                    "scale": scale.tolist(),
                },
                "selection": "rank-pair within annotation type; seeded nearest pool",
                "scope": "discovery-only features; descriptive masks, not a biological null",
            }
        )
    return result
