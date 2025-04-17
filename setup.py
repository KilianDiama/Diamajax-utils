# setup.py
from setuptools import setup, find_packages

setup(
  name="diamajax_utils",
  version="0.1.0",
  packages=find_packages(),          # cherche le dossier diamajax_utils/
  install_requires=[                # si tu as des dépendances obligatoires
    "numpy",
    "onnx",
    "umap-learn",
    "scikit-learn",
    # …
  ],
)
