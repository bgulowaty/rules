from rules.api import Rule, Statement, Relation
from rules.classification.rule_measures import BayesianRuleMeasures


def test_measures_1():
    rule = Rule(
        statements=[Statement(0, Relation.LEQ, threshold=5)], distribution_or_class=0
    )

    x = [[0], [1], [2], [6], [7], [8]]
    y = [0, 0, 1, 0, 1, 0]

    measures = BayesianRuleMeasures.create(rule, x, y)

    assert measures.a() == 2
    assert measures.b() == 2
    assert measures.c() == 1
    assert measures.d() == 1
