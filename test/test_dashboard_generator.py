import pandas as pd
import pytest
from diamajax_utils.dashboard_generator import DashboardGenerator

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "x": [0, 1, 2],
        "y": [10, 20, 30]
    })

def test_generate_static_html(tmp_path, sample_data):
    gen = DashboardGenerator(output_dir=str(tmp_path))
    output_file = gen.generate_static(sample_data, title="Test")
    assert output_file.endswith(".html")
    assert tmp_path.joinpath(output_file).exists()

def test_generate_interactive(tmp_path, sample_data):
    gen = DashboardGenerator(output_dir=str(tmp_path))
    output_file = gen.generate_interactive(sample_data, title="Interact")
    assert output_file.endswith(".html")
    # on peut vérifier qu’il y a un <script> dans le HTML
    content = tmp_path.joinpath(output_file).read_text(encoding="utf-8")
    assert "<script" in content.lower()
