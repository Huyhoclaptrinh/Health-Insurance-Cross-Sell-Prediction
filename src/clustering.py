from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerSegmentation(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        scaled_X = self.scaler.fit_transform(X)
        self.kmeans.fit(scaled_X)
        return self

    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        clusters = self.kmeans.predict(scaled_X)
        return clusters.reshape(-1, 1)