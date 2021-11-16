from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def is_fitted(estimator: BaseEstimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False
