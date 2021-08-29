from typing import Set, Callable, Collection

from attr import attrs, attrib

from rules.api import Rule, Instance
from rules.classification.utils import classify_single_value, to_instance

import numpy as np


@attrs(frozen=True)
class RulesClassifier:
    rules: Set[Rule] = attrib(converter=frozenset)
    classification_strategy: Callable[[Instance, Collection[Rule]], any] = attrib(
        default=classify_single_value
    )

    def fit(self, x, y):
        raise NotImplementedError("aww")

    def predict(self, x):
        y_pred = []
        for inst in x:
            classified_class = self.classification_strategy(
                to_instance(inst), self.rules
            )

            y_pred.append(classified_class)

        return y_pred


def measure_acc(x, y, rules) -> float:
    y_new = RulesClassifier(rules).predict(x)

    correct_classification = sum(
        [1 for y_pred, y_actual in zip(y_new, y) if y_pred == y_actual]
    )

    return correct_classification / len(y)


def measure_metric(x, y, rules, metric) -> float:
    if len(rules) == 0:
        return 0

    y_new = RulesClassifier(rules).predict(x)

    labels = set(np.unique(y))
    get_label_other_than = lambda current_label: list(
        labels.difference({current_label})
    )[0]

    y_new_with_labels = [y if y is not None else get_label_other_than(y) for y in y_new]

    return metric(y, y_new_with_labels)
