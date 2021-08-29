from typing import Dict, Collection

from toolz.curried import pipe, filter, map
from attr import attrs, attrib

from rules.api import Rule, Statement, Relation
from .shrinkage import Rectangle, Bounds


@attrs()
class ShrinkageAdapter:
    rules: Collection[Rule] = attrib()
    default_class: any = attrib()
    rect_by_labeling: Dict[Rectangle, any] = attrib(init=False, factory=dict)

    def as_rectangles(self) -> Collection[Rectangle]:
        rects = []

        for rule in self.rules:
            bounds_by_feature = {}
            for feature, statements in rule.get_statements_by_feature().items():
                sorted_thresholds = pipe(
                    statements, map(lambda s: s.threshold), sorted, list
                )

                if len(sorted_thresholds) != 2:
                    raise Exception("Not supported")

                bounds_by_feature[feature] = Bounds(
                    sorted_thresholds[0], sorted_thresholds[1]
                )

            rect = Rectangle(bounds_by_feature)
            rects.append(rect)
            self.rect_by_labeling[rect] = rule.distribution_or_class

        return rects

    def as_rules(self, rects: Collection[Rectangle]) -> Collection[Rule]:
        rules = []

        for rect in rects:
            statements = []
            for feature, bound in rect.feature_by_bounds.items():
                statements.append(Statement(feature, Relation.MT, bound.lower))
                statements.append(Statement(feature, Relation.LEQ, bound.upper))

            labeling = (
                self.rect_by_labeling[rect]
                if rect in self.rect_by_labeling
                else self.default_class
            )
            rules.append(Rule(statements, labeling))

        return rules
