from collections import defaultdict

from rules.api import Rule, Statement, Relation
from itertools import product


def get_rule_edges(rule: Rule):
    if not rule.is_bounded():
        raise Exception("Only bounded rules supported")

    points_by_feature = defaultdict(list)
    for statement in rule.statements:
        points_by_feature[statement.feature_idx].append(statement.threshold)

    return set(map(tuple, product(*sorted(points_by_feature.values()))))


def get_rule_middles(rule: Rule):
    if not rule.is_bounded():
        raise Exception("Only bounded rules supported")



def test_get_rule_corners():
    r1 = Rule(
        [
            Statement(0, Relation.LEQ, 2),
            Statement(0, Relation.MT, 0),
            Statement(1, Relation.LEQ, 4),
            Statement(1, Relation.MT, 8),
        ],
        0
    )

    edges = get_rule_edges(r1)

    assert {(2, 4), (0, 4), (2, 8), (0, 8)} == edges

def test_get_rule_corners_not_sorted_statements():
    r1 = Rule(
        [
            Statement(1, Relation.LEQ, 4),
            Statement(1, Relation.MT, 8),
            Statement(0, Relation.LEQ, 2),
            Statement(0, Relation.MT, 0),
        ],
        0
    )

    edges = get_rule_edges(r1)

    assert {(2, 4), (0, 4), (2, 8), (0, 8)} == edges
