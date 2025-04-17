import numpy as np
import pytest
from diamajax_utils.clustering_service import ClusteringService

@pytest.fixture
def random_data():
    # 50 points en 5 dimensions
    return np.random.RandomState(0).rand(50, 5)

@pytest.mark.parametrize("algo, kwargs", [
    ("kmeans", {"n_clusters": 3}),
    ("dbscan", {"min_samples": 5}),
])
def test_reduce_and_cluster(random_data, algo, kwargs):
    svc = ClusteringService(
        n_neighbors=5,
        n_components=2,
        clustering_algo=algo,
        **kwargs
    )
    # test réduction
    reduced = svc.reduce_dimensions(random_data)
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (50, 2)

    # test clustering
    labels = svc.apply_clustering(reduced, method=algo, **kwargs)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (50,)
    # pour kmeans, labels entre 0 et n_clusters-1
    if algo == "kmeans":
        assert set(labels) <= set(range(kwargs["n_clusters"]))
