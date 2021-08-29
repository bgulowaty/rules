from typing import List, Tuple

import pytest
from sklearn.model_selection import train_test_split

from .nnge import Example, NNGE, Feature, FeatureType
from sklearn import datasets


def test_it_can_process_one_example():
    clf = NNGE()
    clf.fit([[10, 10]], ["A"])


def test_after_processing_one_example_forms_hyperrectangle():

    clf = NNGE()
    clf.fit([[10, 10]], ["A"])

    assert len(clf.hyperrectangles) == 1


def test_exemplar_covers_example():
    x = [[10, 10]]

    clf = NNGE()
    clf.fit(x, ["A"])
    predicted_label = clf.predict(x)[0]

    assert predicted_label == "A"


def test_given_two_neighbouring_examples_of_same_class_extends_hyperrectangle():
    x, y = [[10, 10], [20, 20]], ["A", "A"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1
    assert clf.predict(y) == y


def creates_correct_hyperrectangle_after_first_sample():
    x, y = [[10, 10]], ["A"]

    clf = NNGE()
    clf.fit(x, y)

    hyperrectangle = list(clf.hyperrectangles)[0]

    assert hyperrectangle.label == y[0]


def test_given_two_different_examples_creates_two_hyperrectangles():
    x, y = [[10, 10], [20, 20]], ["A", "B"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 2
    assert clf.predict(x) == y


def test_given_two_same_class_examples_extends():
    x, y = [[10, 10], [20, 20]], ["A", "A"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1
    assert clf.predict(x) == y


def test_given_example_inbetween_splits_hyperrect():
    x, y = [[10, 10], [10, 20], [10, 15]], ["A", "A", "B"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 3
    assert clf.predict(x) == y


def test_given_example_inbetween_splits_extends_correctly():
    x, y = [[10, 10], [10, 20], [10, 15], [20, 15]], ["A", "A", "A", "A"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1

    clf.fit([[15, 15]], ["B"])

    assert len(clf.hyperrectangles) == 3
    assert clf.predict(x) == y


def test_categorical_learns_single():
    x, y = [["A", "B"]], ["A"]
    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1
    assert clf.predict(x) == y


def test_not_all_dimensions_hyperrect():
    x, y = [[10, 10], [10, 20], [10, 15]], ["A", "A", "B"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 3
    assert clf.predict(x) == y


def test_categorical_two_diff_class_examples_creates_two_hypers():

    x, y = [["A", "B"], ["A", "C"]], ["A", "B"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 2
    assert clf.predict(x) == y


def test_categorical_expands():
    x, y = [["A", "C"], ["A", "B"]], ["A", "A"]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1
    assert clf.predict(x) == y


def test_categorical_dm2011_A():
    x, y = [[0, 3], [2, 0], [2, 1], [2, 4], [4, 2], [5, 3]], [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
    ]
    testing_x_with_generalization = [[x, y] for x in [0, 2, 4, 5] for y in range(0, 4)]
    testing_y_with_generalization = ["A" for _ in testing_x_with_generalization]

    clf = NNGE()
    clf.fit(x, y)

    assert len(clf.hyperrectangles) == 1
    assert clf.predict(testing_x_with_generalization) == testing_y_with_generalization


def test_categorical_dm2011_B():

    x, y = [[0, 3], [2, 0], [2, 1], [2, 4], [4, 2], [5, 3], [2, 3]], [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "B",
    ]
    clf = NNGE()
    clf.fit(x, y)

    assert clf.predict(x) == y


def test_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    clf = NNGE()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = len([1 for y_p, y_act in zip(y_pred, y_test) if y_p == y_act]) / len(y_test)
    print(acc)
    assert acc > 0.9


def create_examples(
    example_values: List[Tuple[List[any], any]], features: List[Feature]
) -> List[Example]:
    return [
        create_example(example_values, features, label)
        for example_values, label in example_values
    ]


def create_example(values: List[any], features: List[Feature], label: any) -> Example:
    return Example(dict(zip(features, values)), label)


def train_all(clf: NNGE, examples: List[Example]):
    [clf.train_single(example) for example in examples]


def assert_correct_class(clf: NNGE, examples: List[Example]):
    for example in examples:
        assert clf.predict_single(example) == example.label
