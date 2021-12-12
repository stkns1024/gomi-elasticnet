from typing import Optional

from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin

from gmelasticnet import elasticnet


class ElasticNet(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        l1: float = 0.1,
        l2: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y) -> "ElasticNet":
        X, y = check_X_y(X, y)

        model = elasticnet.ElasticNet(y, X)
        self._params = model.fit(
            self.l1, self.l2, self.max_iter, self.tol, self.random_state
        )

        return self

    def predict(self, X):
        check_is_fitted(self, "_params")
        X = check_array(X)

        return self._params.predict(X)
