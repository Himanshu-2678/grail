import json
from typing import List, Dict, Any

class DenseRetriever:
    def __init__(self, data_path: str, model_name: str = None):
        # Dense retrieval disabled to fit within Render memory limits (512MB)
        # Bypassing sentence-transformers and torch entirely
        self.catalog = []
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # Returns empty list as dense retrieval is disabled
        return []
