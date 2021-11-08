from typing import Callable, List, Dict, Set

from attr import attrs, attrib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
import pandas as pd


@attrs
class SimpleCompetenceRegionEnsemble(BaseEstimator):
    competence_region_classifier: ClassifierMixin = attrib()
    clf_by_label: Dict[any, BaseEstimator] = attrib(factory=dict)

    def fit(self, x, y):
        x, y = check_X_y(x, y)

        competence_region_labels = self.competence_region_classifier.predict(x)

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

    def predict(self, x):
        x = check_array(x)
        competence_region_labels = self.competence_region_classifier.predict(x)

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
