from typing import List, Dict, Any
from .sparse_retriever import SparseRetriever
from .dense_retriever import DenseRetriever

class HybridRetriever:
    def __init__(self, data_path: str):
        self.sparse_retriever = SparseRetriever(data_path)
        self.dense_retriever = DenseRetriever(data_path)
        
    def _reciprocal_rank_fusion(self, sparse_results: List[Dict], dense_results: List[Dict], k: int = 60) -> List[Dict]:
        rrf_scores = {}
        items = {}
        
        # Process Sparse Results
        for rank, item in enumerate(sparse_results):
            doc_id = item['id']
            items[doc_id] = item
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Process Dense Results
        for rank, item in enumerate(dense_results):
            doc_id = item['id']
            items[doc_id] = item
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_results = []
        for doc_id in sorted_ids:
            item = items[doc_id].copy()
            item['rrf_score'] = rrf_scores[doc_id]
            fused_results.append(item)
            
        return fused_results

    def search(self, query: str, top_k: int = 10, sparse_top_k: int = 60, dense_top_k: int = 0) -> List[Dict[str, Any]]:
        # Dense retrieval is disabled to save memory on Render.
        # We rely on high-quality BM25 retrieval + Symbolic constraint filtering.
        sparse_results = self.sparse_retriever.search(query, top_k=sparse_top_k)
        
        # We map sparse scores to rrf_score for compatibility with the reranker
        for res in sparse_results:
            res['rrf_score'] = res.get('bm25_score', 0.0) / 100.0 # simple scaling
            
        return sparse_results[:top_k]

if __name__ == "__main__":
    retriever = HybridRetriever(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    print("\nTesting Hybrid Retrieval (RRF) with query: 'Java backend'")
    results = retriever.search("Java backend", top_k=3)
    for r in results:
        print(f"[RRF: {r['rrf_score']:.4f}] {r['name']}")
