from typing import Callable, List, Dict, Set

from attr import attrs, attrib
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array
import pandas as pd


@attrs
class SimpleCompetenceRegionEnsemble(BaseEstimator):
    competence_region_classifier: Callable[[any, Set[any]], List[any]] = attrib()
    clf_by_label: Dict[any, BaseEstimator] = attrib(factory=dict)
    competence_regions_used_during_training: List[any] = attrib(factory=list)

    def fit(self, x, y):
        x, y = check_X_y(x, y)

        competence_region_labels = self.competence_region_classifier(x)

        data_by_competence_region = pd.DataFrame(
            {
                "data": x.tolist(),
                "label": y.tolist(),
                "competence": competence_region_labels,
            }
        ).groupby("competence")

        for clf_label, samples in data_by_competence_region:
            self.competence_regions_used_during_training.append(clf_label)

            if clf_label in self.clf_by_label and self.clf_by_label[clf_label] != None:
                self.clf_by_label[clf_label].fit(
                    samples["data"].to_list(), samples["label"].to_list()
                )
            else:
                raise Exception(
                    "Assigned competence region not supported - no classifier exists"
                )

    def predict(self, x):
        x = check_array(x)
        competence_region_labels = self.competence_region_classifier(
            x, self.competence_regions_used_during_training
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
