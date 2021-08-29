import joblib
import networkx as nx
from networkx.algorithms.clique import find_cliques

from rules.api import *
from .measure_adjacencies import measure_rules


def add_to_graph(graph, all_rules, measurements_tuple, measurement):
    if measurement == AdjacentOrNot.NOT_ADJACENT:
        rule_1, rule_2 = measurements_tuple
        rule_idx_1 = all_rules.index(rule_1)
        rule_idx_2 = all_rules.index(rule_2)

        graph.add_node(rule_idx_1)

        graph.add_node(rule_idx_2)

        graph.add_edge(rule_idx_1, rule_idx_2)


def find_non_overlapping_rule_cliques(
    rules, cliques_finding=find_cliques, n_jobs=-1, parallel_backend="threading"
):
    g = nx.Graph()

    with joblib.parallel_backend(parallel_backend):
        all_rule_measurements = measure_rules(rules, n_jobs=n_jobs)

    for measurements_tuple, measurement in all_rule_measurements.items():
        add_to_graph(g, rules, measurements_tuple, measurement)

    return cliques_finding(g)
