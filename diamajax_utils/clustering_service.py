import numpy as np
import logging
from typing import Any, Dict, List, Union

from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
import hdbscan

import matplotlib.pyplot as plt
import plotly.express as px

logger = logging.getLogger(__name__)

class ClusteringService:
    """
    Service avancé pour la réduction de dimensions et le clustering, avec des visualisations interactives.
    """

    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, random_state: int = 42):
        """
        Initialise le service avec des paramètres configurables pour UMAP.

        Args:
            n_neighbors (int): Nombre de voisins pour UMAP.
            min_dist (float): Distance minimale pour UMAP.
            n_components (int): Dimensions cibles pour UMAP (2D ou 3D).
            random_state (int): État aléatoire pour reproductibilité.
        """
        self.reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        logger.info(f"ClusteringService initialized with {n_components}D UMAP reducer.")

    def reduce_dimensions(self, embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Réduit les dimensions des données avec UMAP.

        Args:
            embeddings (Union[List[List[float]], np.ndarray]): Données haute dimension.

        Returns:
            np.ndarray: Données réduites.
        """
        embeddings = self._validate_and_convert_embeddings(embeddings)

        logger.info(f"Reducing dimensions to {self.reducer.n_components}D...")
        try:
            reduced_embeddings = self.reducer.fit_transform(embeddings)
            logger.info("Dimension reduction completed successfully.")
            return reduced_embeddings
        except Exception as e:
            logger.error(f"Error during dimension reduction: {e}")
            raise

    def apply_clustering(self, embeddings: np.ndarray, method: str = "kmeans", **kwargs) -> Dict[str, Any]:
        """
        Applique un clustering sur les données réduites.

        Args:
            embeddings (np.ndarray): Données réduites.
            method (str): Méthode de clustering ('kmeans', 'dbscan', 'hdbscan').
            **kwargs: Paramètres spécifiques à l'algorithme.

        Returns:
            Dict[str, Any]: Résultats avec les labels et le modèle utilisé.
        """
        logger.info(f"Applying clustering using method: {method}...")
        try:
            if method == "kmeans":
                n_clusters = kwargs.get("n_clusters", 5)
                cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == "dbscan":
                eps = kwargs.get("eps", 0.5)
                min_samples = kwargs.get("min_samples", 5)
                cluster_model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            elif method == "hdbscan":
                min_cluster_size = kwargs.get("min_cluster_size", 5)
                cluster_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            labels = cluster_model.fit_predict(embeddings)
            logger.info(f"Clustering completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
            return {"labels": labels, "model": cluster_model}
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise

    def visualize_clusters(self, embeddings: np.ndarray, labels: List[int], interactive: bool = True, save_path: str = None):
        """
        Visualise les clusters en 2D avec des options interactives.

        Args:
            embeddings (np.ndarray): Données réduites en 2D.
            labels (List[int]): Labels des clusters.
            interactive (bool): Générer une visualisation interactive (Plotly) ou statique (Matplotlib).
            save_path (str): Chemin pour enregistrer la visualisation (facultatif).
        """
        if embeddings.shape[1] != 2:
            raise ValueError("Embeddings must be 2D for visualization.")

        logger.info("Generating cluster visualization...")
        try:
            if interactive:
                # Utilisation de Plotly pour une visualisation interactive
                fig = px.scatter(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    color=labels,
                    title="Interactive Cluster Visualization",
                    labels={"x": "UMAP Dim 1", "y": "UMAP Dim 2", "color": "Cluster"},
                )
                if save_path:
                    fig.write_html(save_path)
                    logger.info(f"Interactive cluster visualization saved to {save_path}.")
                fig.show()
            else:
                # Visualisation statique avec Matplotlib
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", s=20)
                plt.title("Static Cluster Visualization")
                plt.xlabel("UMAP Dim 1")
                plt.ylabel("UMAP Dim 2")
                plt.colorbar(scatter, label="Cluster")
                if save_path:
                    plt.savefig(save_path)
                    logger.info(f"Static cluster visualization saved to {save_path}.")
                else:
                    plt.show()
        except Exception as e:
            logger.error(f"Error during cluster visualization: {e}")
            raise

    def cluster_and_visualize(
        self, embeddings: Union[List[List[float]], np.ndarray], method: str = "kmeans", interactive: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Pipeline complet pour réduire les dimensions, clusteriser et visualiser.

        Args:
            embeddings (Union[List[List[float]], np.ndarray]): Données haute dimension.
            method (str): Méthode de clustering ('kmeans', 'dbscan', 'hdbscan').
            interactive (bool): Générer une visualisation interactive ou statique.
            **kwargs: Paramètres spécifiques au clustering.

        Returns:
            Dict[str, Any]: Résultats du clustering.
        """
        logger.info("Starting full clustering pipeline...")
        try:
            reduced_embeddings = self.reduce_dimensions(embeddings)
            clustering_results = self.apply_clustering(reduced_embeddings, method, **kwargs)
            self.visualize_clusters(
                embeddings=reduced_embeddings,
                labels=clustering_results["labels"],
                interactive=interactive,
                save_path=kwargs.get("save_path", None),
            )
            return {"reduced_embeddings": reduced_embeddings, **clustering_results}
        except Exception as e:
            logger.error(f"Error in clustering pipeline: {e}")
            raise

    def _validate_and_convert_embeddings(self, embeddings: Any) -> np.ndarray:
        """
        Valide et convertit les embeddings en numpy array.

        Args:
            embeddings (Any): Données à valider.

        Returns:
            np.ndarray: Données validées et converties.
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.size == 0:
            raise ValueError("Embeddings must be a non-empty 2D numpy array or a list of lists.")
        return embeddings
