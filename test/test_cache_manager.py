import os
import pickle
import pytest
from diamajax_utils.cache_manager import RedisCacheManager as CacheManager

@pytest.fixture
def cache(tmp_path):
    return CacheManager(cache_dir=str(tmp_path))

def test_set_and_get(cache):
    key, value = "foo", {"bar": 123}
    cache.set(key, value)
    assert cache.exists(key)
    loaded = cache.get(key)
    assert loaded == value

def test_delete(cache):
    key, value = "to_del", [1, 2, 3]
    cache.set(key, value)
    cache.delete(key)
   assert not cache.exists(key)
   # get() doit renvoyer None quand la clé a été supprimée
   assert cache.get(key) is None
