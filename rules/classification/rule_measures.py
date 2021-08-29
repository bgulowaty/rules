from typing import Set

import numpy as np
from attr import attrs, attrib

from rules.classification.utils import covered_by_statements, same_label

EPS = np.finfo(float).eps


@attrs()
class BayesianRuleMeasures:
    covered_sample_indicies: Set[int] = attrib()
    same_lable_indicies: Set[int] = attrib()
    all_indicies: Set[int] = attrib()

    @classmethod
    def create(cls, rule, x, y):
        covered_sample_indicies = covered_by_statements(rule, x)
        same_label_indicies = same_label(rule, y)
        all_indicies = set(range(len(x)))

        return cls(covered_sample_indicies, same_label_indicies, all_indicies)

    def a(self) -> float:
        return len(self.same_lable_indicies.intersection(self.covered_sample_indicies))

    def b(self) -> float:
        return len(self.same_lable_indicies.difference(self.covered_sample_indicies))

    def c(self) -> float:
        return len(self.covered_sample_indicies.difference(self.same_lable_indicies))

    def d(self) -> float:
        return len(
            self.all_indicies.difference(self.same_lable_indicies).difference(
                self.covered_sample_indicies
            )
        )

    def s_measure(self) -> float:
        return (self.a() + EPS) / (self.a() + self.c() + EPS) - (self.b() + EPS) / (
            self.b() + self.d() + EPS
        )

    def n_measure(self) -> float:
        return (self.a() + EPS) / (self.a() + self.c() + EPS) - (self.c() + EPS) / (
            self.c() + self.d() + EPS
        )
