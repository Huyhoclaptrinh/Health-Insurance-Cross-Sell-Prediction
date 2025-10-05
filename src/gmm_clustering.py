
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class CustomerSegmentationGMM(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        scaled_X = self.scaler.fit_transform(X)
        self.gmm.fit(scaled_X)
        return self

    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        clusters = self.gmm.predict(scaled_X)
        return clusters.reshape(-1, 1)
