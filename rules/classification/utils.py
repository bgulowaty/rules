from collections import defaultdict
from typing import Collection

from rules.api import Instance, Rule


def classify_single_value(x: Instance, rules: Collection[Rule]) -> any:
    supports = defaultdict(lambda: 0)

    for rule in rules:
        if rule.describes(x):
            supports[rule.classified_class] = supports[rule.classified_class] + 1

    if not supports:
        return None
    return max(supports, key=supports.get)


def to_instance(x: any):
    return Instance(dict(enumerate(x)))


def covered_by_statements(rule, x):
    return {
        idx for idx, x_single in enumerate(x) if rule.describes(to_instance(x_single))
    }


def same_label(rule, y):
    return {
        idx for idx, y_single in enumerate(y) if rule.distribution_or_class == y_single
    }
