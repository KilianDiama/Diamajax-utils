import numpy as np
import pytest
import os

from diamajax_utils.clustering_service import ClusteringService

@pytest.fixture
def sample_embeddings():
    # 50 points en 5 dimensions
    rng = np.random.RandomState(0)
    return rng.rand(50, 5)

def test_validate_and_convert_invalid_input():
    svc = ClusteringService()
    with pytest.raises(ValueError):
        svc._validate_and_convert_embeddings(123)          # pas un array/list
    with pytest.raises(ValueError):
        svc._validate_and_convert_embeddings([1, 2, 3])    # list 1D
    with pytest.raises(ValueError):
        svc._validate_and_convert_embeddings([[]])         # array vide

def test_reduce_dimensions_shape(sample_embeddings):
    svc = ClusteringService(n_neighbors=5, min_dist=0.2, n_components=2, random_state=1)
    reduced = svc.reduce_dimensions(sample_embeddings.tolist())  # accepte list of lists
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (50, 2)

def test_apply_clustering_kmeans_labels_and_model(sample_embeddings):
    svc = ClusteringService()
    reduced = svc.reduce_dimensions(sample_embeddings)
    result = svc.apply_clustering(reduced, method="kmeans", n_clusters=4)
    assert isinstance(result, dict)
    labels = result["labels"]
    model = result["model"]
    assert hasattr(model, "fit_predict")
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (50,)
    assert set(labels) <= set(range(4))

def test_cluster_and_visualize_static(tmp_path, sample_embeddings):
    svc = ClusteringService()
    out = svc.cluster_and_visualize(
        embeddings=sample_embeddings,
        method="kmeans",
        interactive=False,
        save_path=str(tmp_path / "clusters.png"),
        n_clusters=3
    )
    # On doit récupérer reduced_embeddings, labels et model
    assert "reduced_embeddings" in out
    assert "labels" in out and "model" in out
    # Vérifier que le fichier a bien été créé
    assert (tmp_path / "clusters.png").exists()

@pytest.mark.parametrize("method,extra", [
    ("dbscan", {"eps": 0.3, "min_samples": 5}),
    ("hdbscan", {"min_cluster_size": 4})
])
def test_apply_clustering_other_algos(sample_embeddings, method, extra):
    svc = ClusteringService(n_neighbors=10, n_components=2)
    reduced = svc.reduce_dimensions(sample_embeddings)
    res = svc.apply_clustering(reduced, method=method, **extra)
    labels = res["labels"]
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (50,)
