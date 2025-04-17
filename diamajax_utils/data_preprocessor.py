
class DataPreprocessor:
    """
    Gère le prétraitement des données pour le clustering ou les modèles.
    """

    def __init__(self, normalize: bool = True, standardize: bool = True):
        """
        Initialise les paramètres de prétraitement.

        Args:
            normalize (bool): Normaliser les données (MinMax Scaling).
            standardize (bool): Standardiser les données (Moyenne=0, Écart-type=1).
        """
        self.normalize = normalize
        self.standardize = standardize

    def preprocess(self, data: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Applique le prétraitement sur les données.

        Args:
            data (Union[List[List[float]], np.ndarray]): Données brutes.

        Returns:
            np.ndarray: Données prétraitées.
        """
        data = self._validate_and_convert(data)

        if self.standardize:
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        if self.normalize:
            data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

        return data

    def _validate_and_convert(self, data: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Valide et convertit les données en numpy array.

        Args:
            data (Union[List[List[float]], np.ndarray]): Données brutes.

        Returns:
            np.ndarray: Données validées.
        """
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
            raise ValueError("Les données doivent être une liste 2D ou un tableau numpy non vide.")
        return data
