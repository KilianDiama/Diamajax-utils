import os
import pytest
from plotly.graph_objects import Figure
from diamajax_utils.dashboard_generator import DashboardGenerator

@pytest.fixture
def tmp_out(tmp_path):
    # Un dossier temporaire pour les fichiers de sortie
    return str(tmp_path)

def test_create_dashboard(tmp_out):
    data = {
        "ModuleA": {"feature1": 10, "feature2": 20},
        "ModuleB": {"alpha": 5, "beta": 15},
    }
    gen = DashboardGenerator(output_dir=tmp_out)
    output_path = gen.create_dashboard(data, output_file="dashboard.html")

    # Vérifie que le fichier existe et porte le bon nom
    assert os.path.exists(output_path)
    assert output_path.endswith("dashboard.html")

    # Contenu HTML de base
    content = open(output_path, encoding="utf-8").read().lower()
    assert "<html" in content
    assert "dashboard" in content  # titre ou id

def test_generate_sentiment_dashboard(tmp_out):
    sentiment_data = {"positive": 8, "negative": 2, "neutral": 4}
    gen = DashboardGenerator(output_dir=tmp_out)
    output_path = gen.generate_sentiment_dashboard(sentiment_data, output_file="sentiment.html")

    # Vérifie la création du fichier
    assert os.path.exists(output_path)
    assert output_path.endswith("sentiment.html")

    # Doit contenir des éléments liés au sentiment
    content = open(output_path, encoding="utf-8").read()
    assert "sentiment distribution" in content.lower()
    assert "sentiment details" in content.lower()
