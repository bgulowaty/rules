from collections import OrderedDict, defaultdict
from typing import List, Dict

from attr import attrib, attrs
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
from toolz.curried import map, pipe, filter

from loguru import logger

from rules.api import Rule, Instance, Enum
from rules.classification.utils import to_instance


class DepthStrategy(Enum):
    SUBTRACT_FROM_RULES = 1
    ABSOLUTE = 2


@attrs()
class SubspaceRulesClassifier(BaseEstimator):
    rules: List[Rule] = attrib()
    max_depth: int = attrib()
    random_state: int = attrib(default=42)
    train_default_on_whole_set: bool = attrib(default=True)
    depth_strategy: DepthStrategy = attrib(default=DepthStrategy.SUBTRACT_FROM_RULES)
    _clf_by_rule: Dict[Rule, any] = attrib(init=False, factory=dict)
    _default_clf: any = attrib(init=False)

    def fit(self, x, y):
        x, y = check_X_y(x, y)

        already_covered_indicies = set()

        for rule in self.rules:
            if self.depth_strategy == DepthStrategy.SUBTRACT_FROM_RULES:
                depth = self.max_depth - len(rule.statements)
            else:
                depth = self.max_depth

            matching_sample_indicies = pipe(
                x,
                map(to_instance),
                enumerate,
                filter(lambda idx_with_instance: rule.describes(idx_with_instance[1])),
                map(lambda idx_with_instance: idx_with_instance[0]),
                list,
            )

            logger.debug(
                f"Depth={depth}, matching_samples={len(matching_sample_indicies)}"
            )

            if depth >= 1 and len(matching_sample_indicies) >= 1:

                if any(
                    already_covered_indicies.intersection(set(matching_sample_indicies))
                ):
                    raise Exception("Rules are overlapping")

                already_covered_indicies.update(matching_sample_indicies)

                x_train = x[matching_sample_indicies]
                y_train = y[matching_sample_indicies]

                clf = DecisionTreeClassifier(
                    random_state=self.random_state, max_depth=depth
                )
                clf.fit(x_train, y_train)
                self._clf_by_rule[rule] = clf

        not_covered_indices = list(
            set(range(len(x))).difference(already_covered_indicies)
        )
        default_clf = DecisionTreeClassifier(
            random_state=self.random_state, max_depth=self.max_depth
        )
        if len(not_covered_indices) == 0 or self.train_default_on_whole_set:
            default_clf.fit(x, y)
        else:
            x_train = x[not_covered_indices]
            y_train = y[not_covered_indices]
            default_clf.fit(x_train, y_train)

        self._default_clf = default_clf

    def predict(self, x):

        instance_idx_by_rule = defaultdict(list)
        not_covered_indicies = set(range(len(x)))
        for rule in self._clf_by_rule.keys():
            for idx, single_x in enumerate(x):
                if rule.describes(to_instance(single_x)):
                    instance_idx_by_rule[rule].append(idx)
                    not_covered_indicies.remove(idx)

        label_by_idx = OrderedDict()
        for rule, indicies in instance_idx_by_rule.items():
            clf = self._clf_by_rule[rule]
            x_to_classify = x[indicies]

            for label, idx in zip(clf.predict(x_to_classify), indicies):
                label_by_idx[idx] = label

        x_indicies_not_covered = list(not_covered_indicies)

        if len(x_indicies_not_covered) > 0:
            x_not_covered_to_classify = x[x_indicies_not_covered]

            for label, idx in zip(
                self._default_clf.predict(x_not_covered_to_classify),
                x_indicies_not_covered,
            ):
                label_by_idx[idx] = label

        return list(label_by_idx.values())

    def _classify_instance(self, x: List[any]):
        clf = self._find_corresponding_classifier(to_instance(x))

        return clf.predict([x])[0]

    def _find_corresponding_classifier(self, x: Instance) -> any:
        matching_rules = pipe(
            self._clf_by_rule.keys(), filter(lambda r: r.describes(x)), list
        )

        if len(matching_rules) > 1:
            raise Exception("Too many matching rules")

        if len(matching_rules) == 0:
            return self._default_clf

        return self._clf_by_rule[matching_rules[0]]
