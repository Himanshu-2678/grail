import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model_name = model_name
        self.catalog = self._load_data()
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize Embedding Storage (Pure NumPy)
        self.embeddings = None
        
        self._build_index()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _build_index(self):
        corpus_texts = []
        for item in self.catalog:
            taxonomy = item.get("taxonomy", {})
            doc_text = (
                f"Name: {item.get('name', '')}\n"
                f"Assessment Type: {taxonomy.get('assessment_type', '')}\n"
                f"Role Alignment: {', '.join(taxonomy.get('role_alignment', []))}\n"
                f"Description: {item.get('description', '')}"
            )
            corpus_texts.append(doc_text)
            
        print("Encoding corpus for Dense Retrieval...")
        # Encode and store as numpy matrix
        self.embeddings = self.model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
        print(f"Stored {len(self.embeddings)} embeddings in NumPy matrix.")
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        
        # Pure NumPy Cosine Similarity (Dot product on normalized vectors)
        # self.embeddings is (N, D), query_embedding is (1, D)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            item = self.catalog[idx].copy()
            item['dense_score'] = float(scores[idx])
            results.append(item)
                
        return results

if __name__ == "__main__":
    retriever = DenseRetriever(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    print("\nTesting Dense Retrieval with query: 'assessment for a frontend developer using angular'")
    results = retriever.search("assessment for a frontend developer using angular", top_k=3)
    for r in results:
        print(f"[{r['dense_score']:.4f}] {r['name']}")
