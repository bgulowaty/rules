import pytest
import numpy as np
from attr import attrs, attrib
from sklearn.dummy import DummyClassifier

from rules.classification.competence_region_ensemble import (
    SimpleCompetenceRegionEnsemble,
)


@attrs(frozen=True)
class DummyCompetenceRegionClassifier:

    labels = attrib()

    def predict(self, x):
        return np.array(self.labels)


def test_SimpleCompetenceRegionEnsemble_gives_correct_predictions():
    region_assignments = [0, 1, 0, 1, 0, 0]

    ensemble = {
        0: DummyClassifier(strategy="constant", constant="a"),
        1: DummyClassifier(strategy="constant", constant="b"),
    }

    x = [[10], [0], [11], [1], [12], [13]]

    under_test = SimpleCompetenceRegionEnsemble(DummyCompetenceRegionClassifier(region_assignments), ensemble)

    under_test.fit(x, ["a", "b", "a", "b", "a", "a"])

    y_predicted = under_test.predict(x)

    assert y_predicted.tolist() == ["a", "b", "a", "b", "a", "a"]
