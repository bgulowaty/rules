from __future__ import annotations

from collections import Collection
from copy import deepcopy
from itertools import combinations, takewhile
from typing import List, Dict, Union, Tuple
from toolz.curried import pipe, map, filter, reduce
import numpy as np

import logging

from attr import attrs, attrib

EPS = np.finfo(float).eps


@attrs(hash=True, eq=True, auto_attribs=True)
class Bounds:
    lower: float
    upper: float

    def inside(self, other: Bounds) -> bool:
        return self.lower >= other.lower and self.upper <= other.upper


@attrs(hash=False, eq=True)
class Rectangle:
    feature_by_bounds: Dict[int, Bounds] = attrib(factory=dict)

    def covers(self, example: Collection[float]) -> bool:
        covers_for_feature = []
        for x_val, bound in zip(example, self.feature_by_bounds.values()):
            if bound.lower < x_val <= bound.upper:
                covers_for_feature.append(True)
            else:
                covers_for_feature.append(False)

        return all(covers_for_feature)

    def get_size(self) -> float:
        return pipe(
            self.feature_by_bounds.values(),
            map(lambda bound: bound.upper - bound.lower),
            reduce(lambda x, y: x * y),
        )

    def get_feature_by_bounds_range(self) -> Dict[int, float]:
        return {
            feature: bound.upper - bound.lower
            for feature, bound in self.feature_by_bounds.items()
        }

    def is_inside(self, other: Rectangle):
        features = self.feature_by_bounds.keys()
        return all(
            [
                self.feature_by_bounds[feature].inside(other.feature_by_bounds[feature])
                for feature in features
            ]
        )


def calculate_overlapping_area(rect1: Rectangle, rect2: Rectangle) -> float:
    features = rect1.feature_by_bounds.keys()
    area = 0

    for feature in features:
        this_feature_area = 0
        rect1_bounds = rect1.feature_by_bounds[feature]
        rect2_bounds = rect2.feature_by_bounds[feature]

        if (
            rect1_bounds.lower <= rect2_bounds.lower
            and rect1_bounds.upper >= rect2_bounds.upper
        ):
            this_feature_area = rect2_bounds.upper - rect2_bounds.lower
        elif (
            rect2_bounds.lower <= rect1_bounds.lower
            and rect2_bounds.upper >= rect1_bounds.upper
        ):
            this_feature_area = rect1_bounds.upper - rect1_bounds.lower
        elif rect1_bounds.upper >= rect2_bounds.lower >= rect1_bounds.lower:
            this_feature_area = rect1_bounds.upper - rect2_bounds.lower
        elif rect2_bounds.upper >= rect1_bounds.lower >= rect2_bounds.lower:
            this_feature_area = rect2_bounds.upper - rect1_bounds.lower

        if this_feature_area == 0:
            return 0

        area = area * this_feature_area if area != 0 else this_feature_area

    return area


def get_overlapping_size_from_dict(rule1, rule2, overlappings_dict) -> float:
    if (rule1, rule2) in overlappings_dict:
        return overlappings_dict[(rule1, rule2)]

    return overlappings_dict[(rule2, rule1)]


def shrink_rules(rects: Collection[Rectangle], overlapping_limit=0.5):
    rect_combinations = list(combinations(rects, 2))
    rect_combination_with_overlapping_size = pipe(
        rect_combinations,
        map(
            lambda rect_combination: (
                rect_combination,
                calculate_overlapping_area(*rect_combination),
            )
        ),
        dict,
    )

    # delete too big overlappings or nested
    to_delete = set()
    for rect1, rect2 in rect_combinations:
        overlapping_size = get_overlapping_size_from_dict(
            rect1, rect2, rect_combination_with_overlapping_size
        )
        if overlapping_size == 0:
            continue

        rects_with_size = {rect: rect.get_size() for rect in [rect1, rect2]}
        rects_with_overlapping_ratio = {
            rect: overlapping_size / rect.get_size() for rect in [rect1, rect2]
        }

        sorted_by_size = sorted(rects_with_size, key=rects_with_size.get, reverse=True)

        if any(
            [
                ratio >= overlapping_limit
                for ratio in rects_with_overlapping_ratio.values()
            ]
        ):
            to_delete.add(sorted_by_size[0])
            logging.debug(
                f"Deleting rect {sorted_by_size[0]} because of ratio {rects_with_overlapping_ratio.values()}"
            )
        elif rect1.is_inside(rect2) or rect2.is_inside(rect1):
            to_delete.add(sorted_by_size[0])
            logging.debug(
                f"Deleting rect {sorted_by_size[0]} because of its inside/covers another"
            )

    # shrink leftovers
    rects_after_deletion = set(rects).difference(to_delete)
    logging.info(f"Rules to delete size={len(to_delete)}")
    logging.info(f"Rects after deletion={len(rects_after_deletion)}")
    rect_combination_with_overlapping_size = {
        rules_tuple: overlapping_size
        for rules_tuple, overlapping_size in rect_combination_with_overlapping_size.items()
        if rules_tuple[0] in rects_after_deletion
        and rules_tuple[1] in rects_after_deletion
    }

    while any(
        [val for val in rect_combination_with_overlapping_size.values() if val != 0]
    ):
        for rect1, rect2 in rect_combination_with_overlapping_size.keys():
            shrink_on_biggest_feature(rect1, rect2)

        rect_combination_with_overlapping_size = pipe(
            rect_combination_with_overlapping_size,
            map(
                lambda rect_combination: (
                    rect_combination,
                    calculate_overlapping_area(*rect_combination),
                )
            ),
            dict,
        )

    return rects_after_deletion


def extend(rectangles: Collection[Rectangle], X_train) -> Collection[Rectangle]:
    extended_rectangles = list(deepcopy(rectangles))
    for x in X_train:
        if not any([rect.covers(x) for rect in extended_rectangles]):
            candidates = find_possible_extensions(extended_rectangles, x)

            if len(candidates) == 0:
                logging.debug(f"No candidates for {x}")
                extended_rectangles.append(
                    Rectangle(
                        {
                            feature_idx: Bounds(val - EPS, val)
                            for feature_idx, val in enumerate(x)
                        }
                    )
                )
            else:
                candidate_with_size = {
                    candidate: candidate.extended.get_size() for candidate in candidates
                }

                biggest_candidate = max(
                    candidate_with_size, key=candidate_with_size.get
                )

                extended_rectangles.remove(biggest_candidate.original)
                extended_rectangles.append(biggest_candidate.extended)

    return extended_rectangles


@attrs(auto_attribs=True, hash=True, eq=True)
class Candidate:
    original: Rectangle
    extended: Rectangle


def find_possible_extensions(
    rectangles: Collection[Rectangle], single_example
) -> List[Candidate]:
    features = list(range(len(single_example)))
    candidates = []
    for feature in features:
        example_feature_value = single_example[feature]
        for rectangle in rectangles:
            this_rectangle_feature_bounds = deepcopy(rectangle.feature_by_bounds)
            rectangles_without_this_one = set(rectangles).difference({rectangle})

            bounds = rectangle.feature_by_bounds[feature]
            if example_feature_value < bounds.lower:
                this_rectangle_feature_bounds[feature] = Bounds(
                    example_feature_value - EPS, bounds.upper
                )
                possible_candidate = Rectangle(this_rectangle_feature_bounds)

                overlapping_areas = [
                    calculate_overlapping_area(possible_candidate, other_rect)
                    for other_rect in rectangles_without_this_one
                ]
                is_overlapping_with_any = any(
                    [
                        overlapping_area
                        for overlapping_area in overlapping_areas
                        if overlapping_area != 0
                    ]
                )

                if not is_overlapping_with_any:
                    candidates.append(Candidate(rectangle, possible_candidate))
            elif example_feature_value > bounds.upper:
                this_rectangle_feature_bounds[feature] = Bounds(
                    bounds.lower, example_feature_value
                )
                possible_candidate = Rectangle(this_rectangle_feature_bounds)

                overlapping_areas = [
                    calculate_overlapping_area(possible_candidate, other_rect)
                    for other_rect in rectangles_without_this_one
                ]
                is_overlapping_with_any = any(
                    [
                        overlapping_area
                        for overlapping_area in overlapping_areas
                        if overlapping_area != 0
                    ]
                )

                if not is_overlapping_with_any:
                    candidates.append(Candidate(rectangle, possible_candidate))

        return candidates


def shrink_on_biggest_feature(rect1: Rectangle, rect2: Rectangle):
    considered_features = pipe(
        rect1.feature_by_bounds.keys(),
        filter(
            lambda feature: not rect1.feature_by_bounds[feature].inside(
                rect2.feature_by_bounds[feature]
            )
            or not rect2.feature_by_bounds[feature].inside(
                rect1.feature_by_bounds[feature]
            )
        ),
        list,
    )

    if len(considered_features) == 0:
        raise Exception(f"Problem with {rect1} and {rect2}")

    rect1_feature_ranges = rect1.get_feature_by_bounds_range()
    rect2_feature_ranges = rect2.get_feature_by_bounds_range()

    summed_feature_ranegs = {
        feature: rect1_feature_ranges[feature] + rect2_feature_ranges[feature]
        for feature in considered_features
    }

    feature_to_split_on = sorted(
        summed_feature_ranegs, key=summed_feature_ranegs.get, reverse=True
    )[0]

    rect1_bounds = rect1.feature_by_bounds[feature_to_split_on]
    rect2_bounds = rect2.feature_by_bounds[feature_to_split_on]

    if rect1_bounds.inside(rect2_bounds):
        split_point = np.mean([rect1_bounds.lower, rect1_bounds.upper])

        if split_point - rect2_bounds.lower > rect2_bounds.upper - split_point:
            rect2_bounds.upper = split_point
            rect1_bounds.lower = split_point
        else:
            rect2_bounds.lower = split_point
            rect1_bounds.upper = split_point
    elif rect2_bounds.inside(rect1_bounds):
        split_point = np.mean([rect2_bounds.lower, rect2_bounds.upper])

        if split_point - rect1_bounds.lower > rect1_bounds.upper - split_point:
            rect1_bounds.upper = split_point
            rect2_bounds.lower = split_point
        else:
            rect1_bounds.lower = split_point
            rect2_bounds.upper = split_point

    elif rect1_bounds.lower <= rect2_bounds.lower <= rect1_bounds.upper:
        split_range = [rect2_bounds.lower, rect1_bounds.upper]
        split_point = np.mean(split_range)

        rect1_bounds.upper = split_point
        rect2_bounds.lower = split_point

    elif rect2_bounds.lower <= rect1_bounds.lower <= rect2_bounds.upper:
        split_range = [rect1_bounds.lower, rect2_bounds.upper]
        split_point = np.mean(split_range)

        rect1_bounds.lower = split_point
        rect2_bounds.upper = split_point

    if (
        rect1_bounds.upper < rect1_bounds.lower
        or rect2_bounds.upper < rect2_bounds.lower
    ):
        logging.error("WRONG BOUNDS")
        logging.error(rect1_bounds)
        logging.error(rect2_bounds)


def test_calculate_overlapping_area():
    rect1 = Rectangle({1: Bounds(-5, 5)})
    rect2 = Rectangle({1: Bounds(0, 5)})

    assert calculate_overlapping_area(rect1, rect2) == 5


def test_calculate_overlapping_area_2():
    rect1 = Rectangle({1: Bounds(-10, 10)})
    rect2 = Rectangle({1: Bounds(-5, 5)})

    assert calculate_overlapping_area(rect1, rect2) == 10


def test_calculate_overlapping_area_3():
    rect1 = Rectangle({1: Bounds(-5, 5)})
    rect2 = Rectangle({1: Bounds(-10, 10)})

    assert calculate_overlapping_area(rect1, rect2) == 10


def test_calculate_overlapping_area_4():
    rect1 = Rectangle({1: Bounds(0, 5)})
    rect2 = Rectangle({1: Bounds(-5, 5)})

    assert calculate_overlapping_area(rect1, rect2) == 5


def test_calculate_overlapping_area_5():
    rect1 = Rectangle({1: Bounds(-5, 5)})
    rect2 = Rectangle({1: Bounds(20, 30)})

    assert calculate_overlapping_area(rect1, rect2) == 0


def test_calculate_overlapping_2d_area_1():
    rect1 = Rectangle({1: Bounds(-5, 5), 2: Bounds(-5, 0)})
    rect2 = Rectangle({1: Bounds(-5, 5), 2: Bounds(-5, 0)})

    assert calculate_overlapping_area(rect1, rect2) == 50


def test_calculate_overlapping_2d_area_2():
    rect1 = Rectangle({1: Bounds(-10, 0), 2: Bounds(-5, 0)})
    rect2 = Rectangle({1: Bounds(-5, 5), 2: Bounds(-5, 0)})

    assert calculate_overlapping_area(rect1, rect2) == 25


def test_calculate_overlapping_2d_area_3():
    rect1 = Rectangle({1: Bounds(-10, 0), 2: Bounds(-5, 0)})
    rect2 = Rectangle({1: Bounds(-20, -10), 2: Bounds(-5, 0)})

    assert calculate_overlapping_area(rect1, rect2) == 0


def test_inside():
    b1 = Bounds(lower=5.924999952316284, upper=6.125)
    b2 = Bounds(lower=4.3, upper=6.25)

    assert b1.inside(b2) is True
    assert b2.inside(b1) is False


def test_get_size():
    rect1 = Rectangle({1: Bounds(-10, 0), 2: Bounds(0, 5)})

    assert rect1.get_size() == 50


def test_shrink_on_biggest_feature():
    rect1 = Rectangle({1: Bounds(-5, 4)})
    rect2 = Rectangle({1: Bounds(0, 6)})

    shrink_on_biggest_feature(rect1, rect2)

    assert rect1.feature_by_bounds[1] == Bounds(-5, 2)
    assert rect2.feature_by_bounds[1] == Bounds(2, 6)


def test_shrink_on_biggest_feature_2():
    rect1 = Rectangle({1: Bounds(0, 6)})
    rect2 = Rectangle({1: Bounds(-5, 4)})

    shrink_on_biggest_feature(rect1, rect2)

    assert rect2.feature_by_bounds[1] == Bounds(-5, 2)
    assert rect1.feature_by_bounds[1] == Bounds(2, 6)


def test_shrink_on_biggest_feature_3():
    rect1 = Rectangle({1: Bounds(0, 6)})
    rect2 = Rectangle({1: Bounds(3, 6)})

    shrink_on_biggest_feature(rect1, rect2)

    assert rect1.feature_by_bounds[1] == Bounds(0, 4.5)
    assert rect2.feature_by_bounds[1] == Bounds(4.5, 6)


def test_shrink_on_biggest_feature_4():
    rect1 = Rectangle({1: Bounds(0, 10)})
    rect2 = Rectangle({1: Bounds(5, 7)})

    shrink_on_biggest_feature(rect1, rect2)

    assert rect1.feature_by_bounds[1] == Bounds(0, 6)
    assert rect2.feature_by_bounds[1] == Bounds(6, 7)


def test_shrink_only_shrinks():
    rect1 = Rectangle({1: Bounds(0, 6)})
    rect2 = Rectangle({1: Bounds(-5, 4)})

    shrink_rules([rect1, rect2], overlapping_limit=0.8)

    assert rect2.feature_by_bounds[1] == Bounds(-5, 2)
    assert rect1.feature_by_bounds[1] == Bounds(2, 6)


def test_shrink_deletes_containing():
    rect1 = Rectangle({1: Bounds(0, 6)})
    rect2 = Rectangle({1: Bounds(1, 3)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=0.8)

    assert len(new_rules) == 1
    assert rect2 in new_rules


def test_shrink_deletes_smaller_rule_with_too_big_overlapping():
    rect1 = Rectangle({1: Bounds(0, 10)})
    rect2 = Rectangle({1: Bounds(5, 11)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=0.6)

    assert len(new_rules) == 1
    assert rect2 in new_rules


def test_shrink_multiple_rules():
    rect1 = Rectangle({1: Bounds(0, 10)})
    rect2 = Rectangle({1: Bounds(-5, 10)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=0.6)

    assert len(new_rules) == 1
    assert rect1 in new_rules


def test_shrinkage():
    rect1 = Rectangle({1: Bounds(0, 10), 2: Bounds(0, 10)})
    rect2 = Rectangle({1: Bounds(9, 15), 2: Bounds(11, 20)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=0.6)

    assert len(new_rules) == 2
    assert rect1, rect2 in new_rules


def test_shrinkage_2():
    rect1 = Rectangle({1: Bounds(0, 10)})
    rect2 = Rectangle({1: Bounds(1, 11)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=1)

    assert len(new_rules) == 2
    assert rect1.feature_by_bounds == {1: Bounds(0, 5.5)}
    assert rect2.feature_by_bounds == {1: Bounds(5.5, 11)}


def test_shrinkage_3():
    rect1 = Rectangle({1: Bounds(0, 4)})
    rect2 = Rectangle({1: Bounds(-2, 2)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=1)

    assert len(new_rules) == 2
    assert rect1.feature_by_bounds == {1: Bounds(1, 4)}
    assert rect2.feature_by_bounds == {1: Bounds(-2, 1)}


def test_shrinkage_4():
    rect1 = Rectangle({1: Bounds(2, 4)})
    rect2 = Rectangle({1: Bounds(0, 6)})

    new_rules = shrink_rules([rect1, rect2], overlapping_limit=1)

    assert len(new_rules) == 1
    assert rect1 in new_rules


def test_find_possible_candidates():
    rect1 = Rectangle({0: Bounds(2, 4)})
    rect2 = Rectangle({0: Bounds(6, 8)})

    candidates = [c.extended for c in find_possible_extensions([rect1, rect2], [5])]

    assert len(candidates) == 2
    assert Rectangle({0: Bounds(2, 5)}) in candidates
    assert Rectangle({0: Bounds(5, 8)}) in candidates


def test_extend():
    rect1 = Rectangle({0: Bounds(0, 4)})
    rect2 = Rectangle({0: Bounds(6, 8)})

    new_rectangles = extend([rect1, rect2], [[5]])

    assert len(new_rectangles) == 2
    assert Rectangle({0: Bounds(0, 5)}) in new_rectangles
    assert Rectangle({0: Bounds(6, 8)}) in new_rectangles
