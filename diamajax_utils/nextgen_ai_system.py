import asyncio
import logging
from typing import Any, Dict, List

from diamajax_utils.onnx_wrapper import ONNXModelWrapper
from diamajax_utils.dashboard_generator import DashboardGenerator

# Si tu as des modules séparés pour la DB et le NLP, importe-les ici :
# from diamajax_utils.db import PostgreSQLManager, MongoDBManager
# from diamajax_utils.nlp import NLPProcessor, SentimentAnalyzer
# from diamajax_utils.data_organizer import DataOrganizer

logger = logging.getLogger(__name__)

class NextGenAISystem:
    """
    Système AI avancé avec gestion multi-base, NLP, clustering et visualisation.
    """

    def __init__(self, use_postgres: bool = False, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialise le système AI de nouvelle génération.

        Args:
            use_postgres (bool): Définit si PostgreSQL doit être utilisé.
            embedding_model (str): Modèle SentenceTransformers pour les embeddings.
        """
        self.db_manager = (
            PostgreSQLManager(db_name="nextgen_ai", user="postgres", password="securepassword")
            if use_postgres
            else MongoDBManager(db_name="nextgen_ai")
        )
        self.nlp_processor = NLPProcessor(embedding_model=embedding_model)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_organizer = DataOrganizer()
        self.dashboard_generator = DashboardGenerator()
        self.models = {
            "sentiment": ONNXModelWrapper("sentiment_model.onnx"),
            "intent": ONNXModelWrapper("intent_classification.onnx"),
        }
        logger.info("NextGenAISystem initialized.")

    async def process_message(self, user_id: str, user_message: str, language: str = "auto") -> Dict[str, Any]:
        """
        Traite un message utilisateur avec NLP et modèles ONNX.

        Args:
            user_id (str): ID utilisateur.
            user_message (str): Message envoyé par l'utilisateur.
            language (str): Langue cible (détectée automatiquement si "auto").

        Returns:
            Dict[str, Any]: Résultats du traitement (sentiment, intent, réponse).
        """
        try:
            logger.info(f"Processing message for user {user_id}...")

            # Prétraitement du texte
            preprocessed_message = self.nlp_processor.preprocess_text(user_message, target_language="en")

            # Analyse de sentiment avec ONNX
            sentiment_result = self.models["sentiment"].predict({"input_text": preprocessed_message})
            sentiment_label = "positive" if sentiment_result[0] > 0.5 else "negative"

            # Classification d'intention
            intent_result = self.models["intent"].predict({"input_text": preprocessed_message})
            intent_label = "greeting" if intent_result[0] > 0.5 else "request"

            # Réponse générée
            response = (
                f"Sentiment: {sentiment_label} (Confidence: {sentiment_result[0]:.2f}), "
                f"Intent: {intent_label}."
            )

            # Sauvegarder l'interaction
            metadata = {
                "sentiment": {"label": sentiment_label, "confidence": sentiment_result[0]},
                "intent": {"label": intent_label, "confidence": intent_result[0]},
            }
            await self.db_manager.save_interaction(user_id, user_message, response, metadata)
            logger.info(f"Message processed for user {user_id}.")
            return {"response": response, "sentiment": sentiment_label, "intent": intent_label}
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            return {"response": "Error occurred.", "sentiment": None, "intent": None}

    async def process_batch(self, user_id: str, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Traite un lot de messages utilisateur.

        Args:
            user_id (str): ID utilisateur.
            messages (List[str]): Liste des messages utilisateur.

        Returns:
            List[Dict[str, Any]]: Résultats pour chaque message.
        """
        try:
            logger.info(f"Processing batch for user {user_id}...")
            tasks = [self.process_message(user_id, message) for message in messages]
            results = await asyncio.gather(*tasks)
            logger.info(f"Batch processing completed for user {user_id}.")
            return results
        except Exception as e:
            logger.error(f"Error processing batch for user {user_id}: {e}")
            return []

    async def generate_dashboard(self, user_id: str) -> str:
        """
        Génère un tableau de bord des interactions utilisateur.

        Args:
            user_id (str): ID utilisateur.

        Returns:
            str: Chemin vers le tableau de bord HTML généré.
        """
        try:
            interactions = await self.db_manager.fetch_interactions(user_id)
            sentiment_summary = {}
            for interaction in interactions:
                sentiment = interaction["metadata"].get("sentiment", {}).get("label", "unknown")
                sentiment_summary[sentiment] = sentiment_summary.get(sentiment, 0) + 1

            dashboard_path = self.dashboard_generator.generate_sentiment_dashboard(
                sentiment_summary, output_file=f"{user_id}_dashboard.html"
            )
            logger.info(f"Dashboard generated for user {user_id}.")
            return dashboard_path
        except Exception as e:
            logger.error(f"Error generating dashboard for user {user_id}: {e}")
            return ""
