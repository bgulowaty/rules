from __future__ import annotations

from itertools import groupby

from toolz.curried import filter, pipe, map
from enum import Enum
from typing import List, Tuple, Optional, Set, Dict, Collection, Iterable
import numpy as np

from attr import attrs, attrib


@attrs(eq=True, hash=True)
class Example:
    value_by_feature: Dict[Feature, any] = attrib(converter=dict)
    label: Optional[any] = attrib()

    def get_feature_values(self):
        return [
            FeatureValue(feature, value)
            for feature, value in self.value_by_feature.items()
        ]

    def get_feature_value(self, feature: Feature):
        return FeatureValue(feature, self.value_by_feature[feature])


class FeatureType(Enum):
    REAL = 1
    CATEGORICAL = 2


@attrs(frozen=True)
class Feature:
    idx: int = attrib()
    kind: FeatureType = attrib()


@attrs(eq=True, hash=True)
class Statement:
    feature: Feature = attrib()
    lower_boundary: Optional[float] = attrib(factory=lambda: None)
    upper_boundary: Optional[float] = attrib(factory=lambda: None)
    categories: Optional[Set[any]] = attrib(factory=set, converter=frozenset)

    def covers_example(self, example: Example):
        return any([self.covers(fw) for fw in example.get_feature_values()])

    def covers(self, feature_value: FeatureValue) -> bool:
        feature = feature_value.feature

        if feature != self.feature:
            return False

        value = feature_value.value

        if feature.kind == FeatureType.REAL:
            return self.lower_boundary <= value <= self.upper_boundary
        elif feature.kind == FeatureType.CATEGORICAL:
            return value in self.categories

        raise Exception(f"Not supported feature type {feature.kind}")

    def distance_to(self, feature_value: FeatureValue):
        if not self.covers(feature_value):
            raise Exception("Not covered feature")

        if self.feature.kind == FeatureType.CATEGORICAL:
            raise Exception("Distance is not valid for categorical")

        value = feature_value.value

        return (
            self.lower_boundary - value
            if value <= self.lower_boundary
            else value - self.upper_boundary
        )

    def is_categorical(self) -> bool:
        pass

    def is_nominal(self) -> bool:
        pass

    def categorical_values(self) -> Set[any]:
        pass

    def nominal_lower_boundary(self) -> float:
        pass

    def nominal_upper_boundary(self) -> float:
        pass


@attrs(frozen=True)
class Hyperrectangle:
    statements: Set[Statement] = attrib(converter=frozenset)
    label: any = attrib()

    def get_covering_statements(self, example: Example) -> Set[Statement]:
        return pipe(
            self.statements,
            filter(
                lambda statement: any(
                    [statement.covers(fw) for fw in example.get_feature_values()]
                )
            ),
            set,
        )

    def get_statement_by_feature(self) -> Dict[Feature, Statement]:
        return {
            feature: list(statements)[0]
            for feature, statements in groupby(
                sorted(self.statements, key=lambda s: s.feature.idx),
                lambda statement: statement.feature,
            )
        }

    def get_not_covering_statements(self, example: Example):
        return self.statements.difference(self.get_covering_statements(example))

    def covers(self, example: Example) -> bool:
        feature_values = example.get_feature_values()

        inter = [self.any_statement_covers(fv) for fv in feature_values]

        return all(inter)

    def any_statement_covers(self, fv: FeatureValue):
        return any([statement.covers(fv) for statement in self.statements])

    def get_statement_for_feature(self, feature: Feature) -> Optional[Statement]:
        return pipe(
            self.statements,
            filter(lambda statement: statement.feature == feature),
            list,
            lambda statements_list: None
            if len(statements_list) == 0
            else statements_list[0],
        )

    def covers_feature(self, feature: Feature):
        return pipe(
            self.statements, filter(lambda statement: statement.feature == feature), any
        )


@attrs(frozen=True)
class FeatureValue:
    feature: Feature = attrib()
    value: any = attrib()


@attrs(auto_detect=True)
class NNGE:
    examples: Set[Example] = attrib(init=False, factory=set)
    hyperrectangles: Set[Hyperrectangle] = attrib(init=False, factory=set)
    lowest_value_by_feature: Dict[Feature, float] = attrib(factory=dict, init=False)
    highest_value_by_feature: Dict[Feature, float] = attrib(factory=dict, init=False)

    def feature_range(self, feature: Feature) -> float:
        return (
            self.highest_value_by_feature[feature]
            - self.lowest_value_by_feature[feature]
        )

    def add_example(self, example: Example):
        self.update_ranges(example)
        self.examples.add(example)

    def distance_hyper_example(self, rect: Hyperrectangle, example: Example):
        mutual_information = lambda: 1

        to_sum = []
        for feature, value_in_example in example.value_by_feature.items():
            statement_in_hyperrect = rect.get_statement_for_feature(feature)

            if statement_in_hyperrect is None:
                to_sum.append(None)
            elif feature.kind == FeatureType.REAL:
                lower = statement_in_hyperrect.lower_boundary
                upper = statement_in_hyperrect.upper_boundary

                if lower <= value_in_example <= upper:
                    to_sum.append(0)
                elif value_in_example < lower:
                    to_sum.append(
                        (lower - value_in_example) / self.feature_range(feature)
                    )
                elif value_in_example > upper:
                    to_sum.append(
                        (value_in_example - upper) / self.feature_range(feature)
                    )
                else:
                    raise Exception("dupix")

            elif feature.kind == FeatureType.CATEGORICAL:
                if value_in_example in statement_in_hyperrect.categories:
                    to_sum.append(0)
                else:
                    to_sum.append(1)

        non_missing_attributes = pipe(
            to_sum, filter(lambda val: val is not None), list, len
        )

        return (
            np.sqrt(
                sum(
                    [
                        (mutual_information() * dist) ** 2
                        for dist in to_sum
                        if dist is not None
                    ]
                )
            )
            / non_missing_attributes
        )

    @staticmethod
    def infer_feature_type(value):
        return (
            FeatureType.REAL
            if isinstance(value, int) or isinstance(value, float)
            else FeatureType.CATEGORICAL
        )

    @staticmethod
    def convert_to_example(x, y) -> Example:
        value_by_feature = {
            Feature(idx, NNGE.infer_feature_type(val)): val for idx, val in enumerate(x)
        }
        return Example(value_by_feature, y)

    @staticmethod
    def convert_all_to_examples(X, Y) -> List[Example]:
        return [
            NNGE.convert_to_example(x_single, y_single)
            for x_single, y_single in zip(X, Y)
        ]

    def fit(self, x, y):
        examples = NNGE.convert_all_to_examples(x, y)

        for idx, example in enumerate(examples):
            try:
                self.train_single(example)
            except Exception:
                print(f"fucked at {idx} {example}")
                raise Exception()

    def predict(self, x):
        examples = NNGE.convert_all_to_examples(x, [None for _ in x])

        y_new = []
        for example in examples:
            y_new.append(self.predict_single(example))

        return y_new

    def train_single(self, example: Example):
        self.add_example(example)

        closest_hyperrectangle_with_distance = self.find_closest(example)

        if closest_hyperrectangle_with_distance == None:
            self.add_as_new_hyperrectangle(example)
        else:
            ## ADJUST
            closest_hyperrectangle: Hyperrectangle
            distance: float
            (closest_hyperrectangle, distance) = closest_hyperrectangle_with_distance
            if distance == 0:
                if not closest_hyperrectangle.label == example.label:
                    splitted_hyps = self.split(closest_hyperrectangle, example)
                    if splitted_hyps != None:
                        self.hyperrectangles.remove(closest_hyperrectangle)
                        for new_rect in splitted_hyps:
                            self.hyperrectangles.add(new_rect)

            ## TRY GENERALIZE
            self.generalise(example)

            # generalise leftovers
            leftovers = self.find_leftovers()

            while leftovers:
                self.generalise(leftovers[0])
                leftovers = self.find_leftovers()

    def find_leftovers(self):
        return pipe(
            self.examples,
            filter(
                lambda e: not any([rect.covers(e) for rect in self.hyperrectangles])
            ),
            list,
        )

    def generalise(self, example):
        closest_hyperrectangle_with_distance = self.find_closest(example)
        (closest_hyperrectangle, distance) = closest_hyperrectangle_with_distance
        hyperrectangle_candidate = self.extend(closest_hyperrectangle, example)
        covers_conlicting = pipe(
            self.examples,
            filter(lambda e: hyperrectangle_candidate.covers(e)),
            filter(lambda e: e.label != hyperrectangle_candidate.label),
            any,
        )
        if covers_conlicting:
            self.add_as_new_hyperrectangle(example)
        else:
            self.hyperrectangles.remove(closest_hyperrectangle)
            self.hyperrectangles.add(hyperrectangle_candidate)

    def find_closest(self, example: Example) -> Optional[Tuple[Hyperrectangle, float]]:
        if len(self.hyperrectangles) == 0:
            return None

        distance_by_rectangle = {
            rect: self.distance_hyper_example(rect, example)
            for rect in self.hyperrectangles
        }

        minimal_distance_rectangle = min(
            distance_by_rectangle, key=distance_by_rectangle.get
        )

        return (
            (
                minimal_distance_rectangle,
                distance_by_rectangle[minimal_distance_rectangle],
            )
            if minimal_distance_rectangle is not None
            else None
        )

    def add_as_new_hyperrectangle(self, example: Example):
        self.hyperrectangles.add(self.create_exemplar(example))

    @staticmethod
    def create_exemplar(example: Example) -> Hyperrectangle:
        statements = set()
        for feature, value in example.value_by_feature.items():
            if feature.kind == FeatureType.REAL:
                statements.add(
                    Statement(
                        feature=feature, lower_boundary=value, upper_boundary=value
                    )
                )
            elif feature.kind == FeatureType.CATEGORICAL:
                statements.add(Statement(feature=feature, categories={value}))
            else:
                raise Exception("Unsupported feature kind")

        return Hyperrectangle(statements, label=example.label)

    def split(self, rect: Hyperrectangle, example: Example) -> List[Hyperrectangle]:
        examples_with_same_labels_as_rect = pipe(
            self.examples, filter(lambda e: e.label == rect.label), list
        )

        # candidates
        statements_covering_example = pipe(
            rect.statements,
            filter(
                lambda statement: any(
                    [statement.covers(fv) for fv in example.get_feature_values()]
                )
            ),
            list,
        )

        # Check categorical
        categorical_candidates = pipe(
            statements_covering_example,
            filter(lambda statement: statement.feature.kind == FeatureType.CATEGORICAL),
            list,
        )

        categorical_statement_to_split_on = None
        if len(categorical_candidates) > 1:
            categorical_candidates_by_examples_count = {
                statement: self.count_covered_examples(
                    statement, examples_with_same_labels_as_rect
                )
                for statement in statements_covering_example
            }
            categorical_statement_to_split_on = min(
                categorical_candidates_by_examples_count,
                key=categorical_candidates_by_examples_count.get,
            )

        # Check real
        real_candidates = pipe(
            statements_covering_example,
            filter(lambda statement: statement.feature.kind == FeatureType.REAL),
            list,
        )

        real_statements_with_minimal_distance = None
        if len(real_candidates) > 1:
            real_candidates_by_sample_min_distance_to_margin = {
                statement: self.find_min_distance_to_margin(statement, example)
                for statement in real_candidates
            }
            minimal_distance = min(
                real_candidates_by_sample_min_distance_to_margin.values()
            )

            real_statements_with_minimal_distance = pipe(
                real_candidates_by_sample_min_distance_to_margin.items(),
                filter(lambda entry_tuple: entry_tuple[1] == minimal_distance),
                map(lambda entry_tuple: entry_tuple[0]),
                list,
            )

        # TODO DECIDE ON WETHER CAT OR REAL
        if categorical_statement_to_split_on != None:
            return [
                self.prune_hyperrect(rect)
                for rect in self.do_split(
                    rect, categorical_statement_to_split_on, example
                )
            ]

        elif real_statements_with_minimal_distance != None:
            if len(real_statements_with_minimal_distance) > 1:
                best_hyperects = pipe(
                    real_statements_with_minimal_distance,
                    map(lambda s: self.do_split(rect, s, example)),
                    map(
                        lambda rects: (
                            tuple(rects),
                            self.find_largest_number_of_examples_covered(rects),
                        )
                    ),
                    dict,
                    lambda vals: max(vals, key=vals.get),
                )

                return [self.prune_hyperrect(rect) for rect in best_hyperects]

            else:
                return [
                    self.prune_hyperrect(rect)
                    for rect in self.do_split(
                        rect, real_statements_with_minimal_distance[0], example
                    )
                ]

        return None

    def find_largest_number_of_examples_covered(
        self, hyperrects: Iterable[Hyperrectangle]
    ):
        return pipe(
            hyperrects,
            map(lambda rect: len([e for e in self.examples if rect.covers(e)])),
            max,
        )

    def prune_hyperrect(self, rect: Hyperrectangle) -> Hyperrectangle:
        new_statements = set()

        training_examples_covered = pipe(
            self.examples, filter(lambda e: rect.covers(e)), list
        )

        for statement in rect.statements:
            covered_samples = pipe(
                training_examples_covered, filter(statement.covers_example), list
            )
            all_training_data_feature_values = pipe(
                covered_samples,
                map(lambda e: e.value_by_feature[statement.feature]),
                set,
            )
            if statement.feature.kind == FeatureType.CATEGORICAL:
                new_statements.add(
                    Statement(
                        feature=statement.feature,
                        categories=statement.categories.intersection(
                            all_training_data_feature_values
                        ),
                    )
                )

            if statement.feature.kind == FeatureType.REAL:
                lowest_boundary = min(all_training_data_feature_values)
                highest_boundary = max(all_training_data_feature_values)

                new_statements.add(
                    Statement(
                        feature=statement.feature,
                        lower_boundary=lowest_boundary,
                        upper_boundary=highest_boundary,
                    )
                )

        return Hyperrectangle(statements=new_statements, label=rect.label)

    def do_split(
        self, rect: Hyperrectangle, statement: Statement, cause: Example
    ) -> List[Hyperrectangle]:
        examples_covered_by_rect = pipe(
            self.examples, filter(lambda e: rect.covers(e)), list
        )

        rect_statements_without_splitter = rect.statements.difference({statement})

        if statement.feature.kind == FeatureType.CATEGORICAL:
            conflicting_category = cause.value_by_feature[statement.feature]

            categories_without_conflicting = statement.categories.difference(
                {conflicting_category}
            )

            if len(categories_without_conflicting) == 0:
                return [Hyperrectangle(rect_statements_without_splitter, rect.label)]

            new_statement_1 = Statement(
                statement.feature, categories=categories_without_conflicting
            )
            return [
                Hyperrectangle(
                    rect_statements_without_splitter.union({new_statement_1}),
                    label=rect.label,
                )
            ]

        elif statement.feature.kind == FeatureType.REAL:
            cause_feature_value = cause.value_by_feature[statement.feature]

            examples_covered_by_statement = pipe(
                examples_covered_by_rect,
                filter(
                    lambda e: any(
                        [statement.covers(fw) for fw in e.get_feature_values()]
                    )
                ),
                list,
            )

            covered_example_values = pipe(
                examples_covered_by_statement,
                map(lambda example: example.value_by_feature[statement.feature]),
                list,
            )
            first_lower_value = pipe(
                covered_example_values,
                filter(lambda value: value < cause_feature_value),
                max,
            )

            first_higher_value = pipe(
                covered_example_values,
                filter(lambda value: value > cause_feature_value),
                min,
            )

            lower_statements = rect_statements_without_splitter.union(
                {
                    Statement(
                        statement.feature,
                        lower_boundary=statement.lower_boundary,
                        upper_boundary=first_lower_value,
                    )
                }
            )
            upper_statements = rect_statements_without_splitter.union(
                {
                    Statement(
                        statement.feature,
                        upper_boundary=statement.upper_boundary,
                        lower_boundary=first_higher_value,
                    )
                }
            )
            return [
                Hyperrectangle(lower_statements, label=rect.label),
                Hyperrectangle(upper_statements, label=rect.label),
            ]

        raise Exception("Not supported feature type")

    @staticmethod
    def count_covered_examples(
        statement: Statement, examples: Collection[Example]
    ) -> int:
        return sum(
            [
                any([statement.covers(fv) for fv in e.get_feature_values()])
                for e in examples
            ]
        )

    @staticmethod
    def find_min_distance_to_margin(statement: Statement, example: Example) -> float:
        # this will always be one element, no?
        distances = [
            statement.distance_to(fv)
            for fv in example.get_feature_values()
            if statement.covers(fv)
        ]

        return min(distances)

    def extend(self, rect: Hyperrectangle, example: Example) -> Hyperrectangle:
        statements_that_do_not_cover_example = rect.get_not_covering_statements(example)

        expanded_statements = pipe(
            statements_that_do_not_cover_example,
            map(
                lambda statement: self.expand(
                    statement, example.value_by_feature[statement.feature]
                )
            ),
            set,
        )

        new_statements = rect.get_covering_statements(example).union(
            expanded_statements
        )
        return Hyperrectangle(statements=new_statements, label=rect.label)

    def expand(self, statement: Statement, value: any) -> Statement:

        if statement.feature.kind == FeatureType.CATEGORICAL:
            return Statement(
                statement.feature, categories=statement.categories.union({value})
            )
        elif statement.feature.kind == FeatureType.REAL:
            if statement.upper_boundary < value:
                return Statement(
                    statement.feature,
                    lower_boundary=statement.lower_boundary,
                    upper_boundary=value,
                )
            return Statement(
                statement.feature,
                lower_boundary=value,
                upper_boundary=statement.upper_boundary,
            )
        else:
            raise Exception("Not supported feature type")

    def min_value_so_far(self, feature: Feature):
        return pipe(
            self.examples, map(lambda example: example.value_by_feature[feature]), min
        )

    def max_value_so_far(self, feature):
        return pipe(
            self.examples, map(lambda example: example.value_by_feature[feature]), max
        )

    def predict_single(self, example: Example):
        closest_hyperrectangle_with_distance = self.find_closest(example)

        if closest_hyperrectangle_with_distance is not None:
            return closest_hyperrectangle_with_distance[0].label

        return None

    def update_ranges(self, example: Example):
        for feature, value in example.value_by_feature.items():
            if feature.kind == FeatureType.REAL:
                if feature not in self.lowest_value_by_feature:  # or in highest
                    self.lowest_value_by_feature[feature] = value
                    self.highest_value_by_feature[feature] = value
                else:
                    lowest = self.lowest_value_by_feature[feature]
                    highest = self.highest_value_by_feature[feature]

                    if value < lowest:
                        self.lowest_value_by_feature[feature] = value
                    elif value > highest:
                        self.highest_value_by_feature[feature] = value
