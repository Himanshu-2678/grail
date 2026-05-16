import json
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.catalog = self._load_data()
        self.bm25 = None
        self._build_index()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace and punctuation tokenization
        if not text:
            return []
        text = str(text).lower()
        return re.findall(r'\w+', text)
        
    def _build_index(self):
        corpus_tokens = []
        for item in self.catalog:
            # We want to index the name, description, and taxonomy data
            taxonomy = item.get("taxonomy", {})
            
            # Create a rich text representation of the item for BM25
            parts = [
                item.get("name", ""),
                item.get("description", ""),
                " ".join(taxonomy.get("seniority", [])),
                " ".join(taxonomy.get("assessment_type", [])),
                " ".join(taxonomy.get("technical_domain", [])),
                " ".join(taxonomy.get("soft_skills", [])),
                " ".join(taxonomy.get("cognitive_traits", [])),
                " ".join(taxonomy.get("personality_traits", []))
            ]
            
            doc_text = " ".join(parts)
            corpus_tokens.append(self._tokenize(doc_text))
            
        self.bm25 = BM25Okapi(corpus_tokens)
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return if there is some overlap
                item = self.catalog[idx].copy()
                item['bm25_score'] = scores[idx]
                results.append(item)
                
        return results

if __name__ == "__main__":
    retriever = SparseRetriever(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    print("Testing Sparse Retrieval with query: 'Java backend'")
    results = retriever.search("Java backend", top_k=3)
    for r in results:
        print(f"[{r['bm25_score']:.2f}] {r['name']} - {r['taxonomy']['technical_domain']}")
