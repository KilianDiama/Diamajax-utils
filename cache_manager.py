class RedisCacheManager:
    """
    Gère un cache distribué avec Redis et fallback local.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, cache_dir: str = "cache", ttl: int = 3600):
        """
        Initialise le gestionnaire de cache.

        Args:
            redis_host (str): Adresse du serveur Redis.
            redis_port (int): Port du serveur Redis.
            cache_dir (str): Répertoire pour le fallback local.
            ttl (int): Temps de vie des entrées en secondes.
        """
        self.redis = Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        self.local_cache = TTLCache(maxsize=1000, ttl=ttl)
        logger.info(f"RedisCacheManager initialized. Redis at {redis_host}:{redis_port}, fallback in '{cache_dir}'.")

    def redis_available(self) -> bool:
        """
        Vérifie si Redis est accessible.

        Returns:
            bool: True si Redis est disponible, sinon False.
        """
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key (str): Clé de l'entrée dans le cache.

        Returns:
            Optional[Any]: Valeur associée ou None si non trouvée.
        """
        # Vérifier Redis
        if self.redis_available():
            try:
                value = self.redis.get(key)
                if value:
                    logger.info(f"Cache hit in Redis for key: {key}")
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Error fetching key '{key}' from Redis: {e}")

        # Vérifier le cache local
        if key in self.local_cache:
            logger.info(f"Cache hit in local memory for key: {key}")
            return self.local_cache[key]

        # Vérifier le fallback local
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    logger.info(f"Cache hit in local fallback for key: {key}")
                    return json.load(file)
            except Exception as e:
                logger.error(f"Error reading local cache file for key '{key}': {e}")

        logger.info(f"Cache miss for key: {key}")
        return None

    def set_cache(self, key: str, value: Any):
        """
        Ajoute une valeur au cache.

        Args:
            key (str): Clé de l'entrée.
            value (Any): Valeur associée à la clé.
        """
        # Ajouter dans Redis
        if self.redis_available():
            try:
                self.redis.setex(key, self.ttl, json.dumps(value))
                logger.info(f"Cache set in Redis for key: {key}")
            except Exception as e:
                logger.error(f"Error setting key '{key}' in Redis: {e}")

        # Ajouter dans le cache local
        self.local_cache[key] = value

        # Ajouter dans le fallback local
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(file_path, "w") as file:
                json.dump(value, file)
            logger.info(f"Cache set in local fallback for key: {key}")
        except Exception as e:
            logger.error(f"Error writing local cache file for key '{key}': {e}")

    def delete_cache(self, key: str):
        """
        Supprime une entrée du cache.

        Args:
            key (str): Clé de l'entrée à supprimer.
        """
        # Supprimer de Redis
        if self.redis_available():
            try:
                self.redis.delete(key)
                logger.info(f"Cache key '{key}' deleted from Redis.")
            except Exception as e:
                logger.error(f"Error deleting key '{key}' from Redis: {e}")

        # Supprimer du cache local
        if key in self.local_cache:
            del self.local_cache[key]
            logger.info(f"Cache key '{key}' deleted from local memory.")

        # Supprimer du fallback local
        file_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cache file '{key}.json' deleted from local fallback.")
            except Exception as e:
                logger.error(f"Error deleting cache file '{key}.json': {e}")

    def clear_cache(self):
        """
        Supprime tout le cache, dans Redis et localement.
        """
        # Effacer Redis
        if self.redis_available():
            try:
                self.redis.flushdb()
                logger.info("All Redis cache cleared.")
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")

        # Effacer le cache local
        self.local_cache.clear()
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            logger.info("All local fallback cache cleared.")
        except Exception as e:
            logger.error(f"Error clearing local cache: {e}")
