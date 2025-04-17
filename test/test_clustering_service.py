import numpy as np
import pytest
from diamajax_utils.clustering_service import ClusteringService

def test_importable_and_core_methods():
    assert hasattr(ClusteringService, "reduce_dimensions")
    assert hasattr(ClusteringService, "apply_clustering")

@pytest.mark.parametrize("algo,kargs", [
    ("kmeans", {"n_clusters": 2}),
    ("dbscan", {"min_samples": 2}),
])
def test_reduce_and_cluster_shape(algo, kargs):
    data = np.random.rand(50, 10)
    svc = ClusteringService(n_neighbors=5, n_components=2, clustering_algo=algo, **kargs)
    reduced = svc.reduce_dimensions(data)
    assert reduced.shape == (50, 2)

    labels = svc.apply_clustering(reduced, method=algo, **kargs)
    assert len(labels) == 50
    # Pour kmeans, labels ∈ {0,1}
    if algo == "kmeans":
        assert set(labels) <= set(range(2))
