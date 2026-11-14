"""Model wrappers that must be importable wherever artifacts are unpickled."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


class LogTargetRegressor:
    """Fit on log1p(y), predict expm1. Skewed resolution times make the median-ish
    log-space fit a better MAE minimiser than a mean-squared fit on raw days."""

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.pipe.predict(X)), 0, None)


class SafeSVD(TruncatedSVD):
    """TruncatedSVD that caps n_components below the vocabulary size (tiny corpora)."""

    def fit_transform(self, X, y=None):
        self.n_components = max(1, min(self.n_components, X.shape[1] - 1))
        return super().fit_transform(X, y)
