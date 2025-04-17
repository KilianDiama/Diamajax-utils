import os
import io
import logging
from typing import Dict

from selenium import webdriver
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Génère des tableaux de bord interactifs avec support pour l’exportation.
    """

    def __init__(self, output_dir: str = "dashboards"):
        """
        Initialise la classe DashboardGenerator.

        Args:
            output_dir (str): Répertoire pour sauvegarder les tableaux de bord.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"DashboardGenerator initialized. Output directory: {output_dir}")

    def create_dashboard(
        self, data: Dict[str, Dict[str, int]], output_file: str = "dashboard.html"
    ) -> str:
        """
        Crée un tableau de bord interactif basé sur les données fournies.

        Args:
            data (Dict[str, Dict[str, int]]): Données pour générer les graphiques.
            output_file (str): Nom du fichier HTML exporté.

        Returns:
            str: Chemin vers le fichier généré.
        """
        try:
            logger.info("Creating dashboard...")
            fig = make_subplots(
                rows=1,
                cols=len(data),
                subplot_titles=[f"{key}" for key in data.keys()],
            )

            # Ajout des graphiques
            col = 1
            for title, values in data.items():
                fig.add_trace(
                    go.Bar(x=list(values.keys()), y=list(values.values()), name=title),
                    row=1,
                    col=col,
                )
                col += 1

            fig.update_layout(
                title="Dashboard",
                barmode="group",
                template="plotly_dark",
                height=600,
                width=1200,
            )

            # Exporter le tableau de bord
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            logger.info(f"Dashboard exported to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return ""

    def add_pie_chart(self, fig: go.Figure, data: Dict[str, int], title: str, row: int, col: int):
        """
        Ajoute un graphique en camembert au tableau de bord.

        Args:
            fig (go.Figure): Figure Plotly existante.
            data (Dict[str, int]): Données pour le camembert.
            title (str): Titre du graphique.
            row (int): Ligne cible.
            col (int): Colonne cible.
        """
        try:
            fig.add_trace(
                go.Pie(labels=list(data.keys()), values=list(data.values()), title=title),
                row=row,
                col=col,
            )
            logger.info(f"Pie chart added for: {title}")
        except Exception as e:
            logger.error(f"Error adding pie chart: {e}")

    def generate_sentiment_dashboard(
        self, sentiment_data: Dict[str, int], output_file: str = "sentiment_dashboard.html"
    ) -> str:
        """
        Génère un tableau de bord pour l'analyse des sentiments.

        Args:
            sentiment_data (Dict[str, int]): Données d’analyse des sentiments.
            output_file (str): Nom du fichier exporté.

        Returns:
            str: Chemin vers le fichier généré.
        """
        try:
            logger.info("Generating sentiment analysis dashboard...")
            fig = make_subplots(
                rows=1, cols=2, subplot_titles=["Sentiment Distribution", "Sentiment Details"]
            )

            # Ajouter un graphique en barre
            self.add_pie_chart(fig, sentiment_data, "Sentiment Distribution", row=1, col=1)

            # Ajouter un graphique détaillé
            fig.add_trace(
                go.Bar(x=list(sentiment_data.keys()), y=list(sentiment_data.values()), name="Details"),
                row=1,
                col=2,
            )

            # Exporter le tableau de bord
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            logger.info(f"Sentiment dashboard exported to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating sentiment dashboard: {e}")
            return ""

    def export_to_image(self, input_file: str, output_file: str):
        """
        Exporte un tableau de bord HTML en image (PNG).

        Args:
            input_file (str): Chemin du fichier HTML d’entrée.
            output_file (str): Chemin du fichier image exporté.
        """
        try:
            from selenium import webdriver
            from PIL import Image

            # Charger le fichier HTML avec Selenium
            driver = webdriver.Chrome()
            driver.get(f"file://{os.path.abspath(input_file)}")
            screenshot = driver.get_screenshot_as_png()
            driver.quit()

            # Sauvegarder comme image
            image = Image.open(io.BytesIO(screenshot))
            image.save(output_file)
            logger.info(f"Dashboard exported to image: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting dashboard to image: {e}")
