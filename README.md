# diamajax-utils

[![PyPI version](https://img.shields.io/pypi/v/diamajax-utils)](https://pypi.org/project/diamajax-utils) [![Build Status](https://github.com/KilianDiama/Diamajax-utils/workflows/CI/badge.svg)](https://github.com/KilianDiama/Diamajax-utils/actions) [![License](https://img.shields.io/github/license/KilianDiama/Diamajax-utils)](LICENSE)

**A Python package providing key AI utilities for production-ready workflows**:

- **ONNXModelWrapper**: simplified multi-device ONNX inference with warmup and input validation.
- **ClusteringService**: UMAP dimensionality reduction + clustering (KMeans, DBSCAN, HDBSCAN) + interactive visualization.
- **NextGenAISystem**: end-to-end AI pipeline (NLP preprocessing, ONNX inference, caching, storage, dashboards).

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



