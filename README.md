Diamajax-utils 🚀

Une boîte à outils Python performante et modulaire conçue pour accélérer la mise en production de vos modèles d'intelligence artificielle.

💡 Pourquoi Diamajax-utils ?

Diamajax-utils simplifie le processus souvent complexe du déploiement d'IA grâce à des modules pré-conçus, performants et faciles à intégrer.

🚀 Fonctionnalités clés

✅ Inférence ONNX

Inférence rapide et optimisée

Gestion multi-device (CPU/GPU)

Warm-up intégré pour performances optimales

Validation automatique des modèles

📊 Clustering avancé

Intégration efficace de UMAP

Algorithmes variés : KMeans, DBSCAN, HDBSCAN

Visualisation rapide et intuitive

📈 Dashboards interactifs

Génération automatique

Compatible Streamlit et Dash

Prêt à l'emploi, personnalisable en quelques clics

⚡ Gestion du Cache et Prétraitement

Accélération de workflows via cache intelligent

Prétraitement simplifié et modulaire

Réutilisation facile des transformations de données

📦 Installation

Installation facile avec PyPI :

pip install diamajax-utils

🚨 Exemples d'utilisation

from diamajax_utils.inference import ONNXInference

# Chargement et inférence d'un modèle ONNX
model = ONNXInference(model_path="model.onnx", device="cuda")
result = model.predict(input_data)

Plus d'exemples dans le dossier examples/.

🛠️ Contribution

Votre contribution est bienvenue !

Forkez ce dépôt

Créez une branche de fonctionnalité (git checkout -b feature/AmazingFeature)

Commitez vos modifications (git commit -m 'Add some AmazingFeature')

Poussez votre branche (git push origin feature/AmazingFeature)

Ouvrez une pull request

📖 Documentation

Consultez la documentation complète ici.

🧪 Tests

Exécutez les tests avec Pytest :

pytest tests/

📄 Licence

Distribué sous licence . Voir LICENSE V1 pour plus d'informations.

✨ Contacts

Kilian Diama : GitHub

⭐️ Si vous appréciez ce projet, une étoile ⭐️ sur GitHub serait très appréciée !
