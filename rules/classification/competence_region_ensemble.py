from __future__ import annotations
from copy import deepcopy
from typing import Callable, List, Dict, Set
from toolz.curried import pipe, map, filter
import numpy as np

from attr import attrs, attrib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array

from .sklearn_utils import is_fitted
import pandas as pd

# todo(bgulowaty): add type for competence region classifier

@attrs
class SimpleCompetenceRegionEnsemble(BaseEstimator):
    competence_region_classifier: BaseEstimator = attrib() # for each .predict sample, should return list of all competence zones, sorted by distance
    clf_by_label: Dict[any, BaseEstimator] = attrib(factory=dict)

    def fit(self, x, y, competence_region_classifier=None):
        if competence_region_classifier != None:
            self.competence_region_classifier = competence_region_classifier
        else:
            raise Exception("No competence region classifier was given")

        x, y = check_X_y(x, y)

        competence_region_labels = SimpleCompetenceRegionEnsemble.get_only_first_ones(self.competence_region_classifier.predict(x))

        data_by_competence_region = pd.DataFrame(
            {
                "data": x.tolist(),
                "label": y.tolist(),
                "competence": competence_region_labels.tolist(),
            }
        ).groupby("competence")

        for clf_label, samples in data_by_competence_region:

            if clf_label in self.clf_by_label and self.clf_by_label[clf_label] is not None:
                self.clf_by_label[clf_label].fit(
                    samples["data"].to_list(), samples["label"].to_list()
                )
            else:
                raise Exception(
                    "Assigned competence region not supported - no classifier exists"
                )

        return self

    @staticmethod
    def get_only_first_ones(list_of_lists):
        return pipe(list_of_lists,
             map(lambda labels: labels[0]),
             list,
             np.array
            )

    def get_first_trained_competence_region(self, labels):
        return pipe(labels,
                    filter(lambda label: is_fitted(self.clf_by_label[label])),
                    list,
                    lambda the_labels: the_labels[0]
                    )

    def predict(self, x):
        x = check_array(x)
        competence_region_labels = pipe(
            self.competence_region_classifier.predict(x),
            map(self.get_first_trained_competence_region),
            list
        )

        return (
            pd.DataFrame({"data": x.tolist(), "competence": competence_region_labels})
            .groupby("competence")["data"]
            .transform(
                lambda samples: self.clf_by_label[samples.name].predict(
                    samples.to_list()
                )
            )
            .to_numpy()
        )
