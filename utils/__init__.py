from .get_predicates import get_predicates
from .loaders import load_clusters
from .embedder import Embedder
from .call_llm import call_llm
from .smart_list import SmartList

def set_encoder(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError