from typing import List

from rules.api import Rule, Statement, Relation
from ..nnge.nnge import Hyperrectangle, Statement as HyperrectStatement, Feature as HyperrectFeature, FeatureType, \
    Example as NNGEExample, NNGE
from toolz.curried import pipe, map, filter
import numpy as np

from ..classification.classification import to_instance
from .utils import EPS


def to_nnge_hyper(rule: Rule, x) -> Hyperrectangle:
    hyperrect_statemets = set()
    for feature_idx, statements in rule.get_statements_by_feature().items():
        if len(statements) != 2:
            raise Exception("nah")
        sorted_thresholds = pipe(statements,
                                 map(lambda s: s.threshold),
                                 sorted,
                                 list)
        feature = HyperrectFeature(feature_idx, FeatureType.REAL)
        hyperrect_statemets.add(
            HyperrectStatement(feature, lower_boundary=sorted_thresholds[0]-EPS, upper_boundary=sorted_thresholds[1]))

    hyper = Hyperrectangle(hyperrect_statemets, label=rule.classified_class)
    return hyper


def to_rule(hyperrect: Hyperrectangle) -> Rule:
    rule_statements = set()

    for feature, statement in hyperrect.get_statement_by_feature().items():
        if statement.upper_boundary != np.inf and statement.lower_boundary != -np.inf:
            rule_statements.add(Statement(feature.idx, Relation.LEQ, statement.upper_boundary))
            rule_statements.add(Statement(feature.idx, Relation.MT, statement.lower_boundary-EPS))

    return Rule(list(rule_statements), hyperrect.label)

def toNNGEExample(x_val, y_val):
    return NNGEExample({
        HyperrectFeature(idx, FeatureType.REAL): val for idx, val in enumerate(x_val)
    }, y_val)

def train_nnge_and_get_rules(seed: List[Rule], x_train, y_train):
    feature_maxes = np.max(x_train, axis=0)
    feature_lowest = np.min(x_train, axis=0)

    covered_sample_indicies = pipe(
        x_train,
        enumerate,
        filter(lambda idx_with_x: any([r.describes(to_instance(idx_with_x[1])) for r in seed])),
        map(lambda idx_with_x: idx_with_x[0]),
        set)

    x_seed = x_train[list(covered_sample_indicies)]
    y_seed = y_train[list(covered_sample_indicies)]

    seed_rules = [r for r in seed if any([r.describes(to_instance(e)) for e in x_seed])]
    seed_hyper = [to_nnge_hyper(rule, x_seed) for rule in seed_rules]
    lowest_value_by_feature = {
        HyperrectFeature(idx, FeatureType.REAL): val for idx, val in enumerate(feature_lowest)
    }
    highest_value_by_feature = {
        HyperrectFeature(idx, FeatureType.REAL): val for idx, val in enumerate(feature_maxes)
    }
    seed_examples = [
        toNNGEExample(x_val, y_val) for x_val, y_val in zip(x_seed, y_seed)
    ]

    clf = NNGE()
    clf.examples = set(seed_examples)
    clf.hyperrectangles = set(seed_hyper)
    clf.lowest_value_by_feature = lowest_value_by_feature
    clf.highest_value_by_feature = highest_value_by_feature

    not_covered_example_indicies = set(range(len(x_train))).difference(covered_sample_indicies)

    x_to_train_nnge = x_train[list(not_covered_example_indicies)]
    y_to_train_nnge = y_train[list(not_covered_example_indicies)]


    clf.fit(x_to_train_nnge, y_to_train_nnge)

    return [to_rule(hyper) for hyper in clf.hyperrectangles]