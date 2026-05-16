import json
import os
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, data_path: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.data_path = data_path
        self.model_name = model_name
        self.catalog = self._load_data()
        
        # Initialize embedding model
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity (assuming normalized vectors)
        
        self._build_index()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _build_index(self):
        corpus_texts = []
        for item in self.catalog:
            # For dense retrieval, we embed a descriptive string
            taxonomy = item.get("taxonomy", {})
            
            # Use specific formatting for BGE models (often benefit from clear structure)
            doc_text = (
                f"Name: {item.get('name', '')}\n"
                f"Assessment Type: {taxonomy.get('assessment_type', '')}\n"
                f"Role Alignment: {', '.join(taxonomy.get('role_alignment', []))}\n"
                f"Description: {item.get('description', '')}"
            )
            corpus_texts.append(doc_text)
            
        # Encode corpus
        print("Encoding corpus for Dense Retrieval...")
        embeddings = self.model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=False)
        
        # Add to FAISS
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"Added {self.index.ntotal} items to FAISS index.")
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # BGE models typically suggest a specific prefix for retrieval queries, 
        # though v1.5 handles prefix-less better, it's good practice.
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        
        scores, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # valid index
                item = self.catalog[idx].copy()
                item['dense_score'] = float(scores[0][i])
                results.append(item)
                
        return results

if __name__ == "__main__":
    retriever = DenseRetriever(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    print("\nTesting Dense Retrieval with query: 'assessment for a frontend developer using angular'")
    results = retriever.search("assessment for a frontend developer using angular", top_k=3)
    for r in results:
        print(f"[{r['dense_score']:.4f}] {r['name']}")
