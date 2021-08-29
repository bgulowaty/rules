import pytest
from sklearn.dummy import DummyClassifier

from rules.classification.competence_region_ensemble import (
    SimpleCompetenceRegionEnsemble,
)


@pytest.mark.skip(reason="broken test")
def test_SimpleCompetenceRegionEnsemble_gives_correct_predictions():
    region_assignments = [0, 1, 0, 1, 0, 0]
    region_assigning_function = lambda x: region_assignments

    ensemble = {
        0: DummyClassifier(strategy="constant", constant="a"),
        1: DummyClassifier(strategy="constant", constant="b"),
    }

    x = [[10], [0], [11], [1], [12], [13]]

    under_test = SimpleCompetenceRegionEnsemble(region_assigning_function, ensemble)

    under_test.fit(x, ["a", "b", "a", "b", "a", "a"])

    y_predicted = under_test.predict(x)

    assert y_predicted.tolist() == ["a", "b", "a", "b", "a", "a"]
