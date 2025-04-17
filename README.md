⚙️ Diamajax-utils — Build AI like a Pro
Unleash powerful AI pipelines with zero hassle.

Diamajax-utils est une boîte à outils Python pensée pour les devs et data scientists qui veulent aller au-delà des POCs.
Passe direct en production-ready mode avec des modules intelligents pour :

🔥 Inférence ONNX optimisée (multi-device, warmup, validation)

🧠 Clustering avancé (UMAP + KMeans/DBSCAN/HDBSCAN)

📊 Dashboards interactifs auto-générés

⚡ Cache & preprocessing pour des workflows fluides

🛠️ Plug & play AI pipelines – tout est modulaire, tout est scalable

Tu veux livrer plus vite, mieux, et sans te battre avec les détails techniques ?
Diamajax-utils te file les clés.



---

## Installation

Install from PyPI:

```bash
pip install diamajax-utils
```

Or clone and install locally:

```bash
git clone https://github.com/KilianDiama/Diamajax-utils.git
cd Diamajax-utils
pip install .
```

---

## Quickstart

### ONNXModelWrapper

```python
from diamajax_utils import ONNXModelWrapper

# Load model on GPU if available
wrapper = ONNXModelWrapper("path/to/model.onnx", device_preference="auto")
# Inspect inputs/outputs
print(wrapper.get_model_metadata())
# Warm up
wrapper.warmup({"input": dummy_array})
# Run inference
outputs = wrapper.predict({"input": input_array})
```

### ClusteringService

```python
from diamajax_utils import ClusteringService

clust = ClusteringService(n_neighbors=15, min_dist=0.1)
# Reduce high-dimensional embeddings
reduced = clust.reduce_dimensions(embeddings)
# Cluster and visualize
result = clust.apply_clustering(reduced, method="hdbscan", min_cluster_size=5)
```

### NextGenAISystem

```python
from diamajax_utils import NextGenAISystem

system = NextGenAISystem(use_postgres=False)
# Process a single message
res = await system.process_message("user1", "Hello, how are you?")
print(res["response"])
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Run tests: `pytest`.
5. Push and open a Pull Request.

📄 Licence

Ce projet est distribué sous la Diamajax License v1.0 © 2025 Matthieu Ouvrard (aka Diamajax).
Pour usage commercial, contactez l’auteur : diamajax@gmail.com.

Voir le fichier LICENSE pour le texte complet.



