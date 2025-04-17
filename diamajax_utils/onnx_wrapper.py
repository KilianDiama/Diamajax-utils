import logging
from typing import Any, Dict, List

import onnxruntime as ort

logger = logging.getLogger(__name__)

class ONNXModelWrapper:
    """
    Encapsulation pour les modèles ONNX avec support multi-device et gestion des erreurs.
    """

    def __init__(self, model_path: str, device_preference: str = "auto"):
        """
        Initialise la classe avec un chemin de modèle ONNX.

        Args:
            model_path (str): Chemin vers le fichier ONNX.
            device_preference (str): Préférence de device ('cpu', 'gpu', ou 'auto').
        """
        self.model_path = model_path
        self.device = self._select_device(device_preference)
        self.session = ort.InferenceSession(model_path, providers=[self.device])
        self.input_metadata = {input.name: input.shape for input in self.session.get_inputs()}
        self.output_metadata = {output.name: output.shape for output in self.session.get_outputs()}
        logger.info(f"Model loaded from {model_path} on device: {self.device}")

    def _select_device(self, preference: str) -> str:
        """
        Sélectionne le device en fonction de la préférence et de la disponibilité.

        Args:
            preference (str): Préférence de device ('cpu', 'gpu', ou 'auto').

        Returns:
            str: Device sélectionné.
        """
        if preference == "gpu" or (preference == "auto" and ort.get_device() == "GPU"):
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Valide les données d'entrée pour le modèle ONNX.

        Args:
            input_data (Dict[str, Any]): Données d'entrée.

        Returns:
            bool: True si les données sont valides, sinon False.
        """
        for key, value in input_data.items():
            if key not in self.input_metadata:
                logger.error(f"Invalid input key: {key}. Expected keys: {list(self.input_metadata.keys())}")
                return False
            if len(value.shape) != len(self.input_metadata[key]):
                logger.error(f"Shape mismatch for input '{key}': {value.shape} != {self.input_metadata[key]}")
                return False
        logger.info("Input validation passed.")
        return True

    def predict(self, input_data: Dict[str, Any]) -> List[Any]:
        """
        Effectue une prédiction avec le modèle ONNX.

        Args:
            input_data (Dict[str, Any]): Données d'entrée au modèle.

        Returns:
            List[Any]: Résultats de la prédiction.
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data provided.")

        try:
            logger.info(f"Running inference on model: {self.model_path}")
            outputs = self.session.run(None, input_data)
            logger.info("Inference completed successfully.")
            return outputs
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return []

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Récupère les métadonnées du modèle ONNX.

        Returns:
            Dict[str, Any]: Métadonnées incluant les entrées et sorties.
        """
        metadata = {
            "inputs": self.input_metadata,
            "outputs": self.output_metadata,
        }
        logger.info(f"Model metadata: {metadata}")
        return metadata

    def warmup(self, sample_input: Dict[str, Any]):
        """
        Réalise une pré-exécution pour réduire la latence initiale.

        Args:
            sample_input (Dict[str, Any]): Exemple de données d'entrée.
        """
        try:
            logger.info("Warming up ONNX model...")
            self.predict(sample_input)
            logger.info("Warmup completed successfully.")
        except Exception as e:
            logger.error(f"Error during warmup: {e}")
