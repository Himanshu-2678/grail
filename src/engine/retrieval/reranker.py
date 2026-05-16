from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str = None):
        # Disabled CrossEncoder to fit within Render's memory limits (512MB)
        self.model = None
        
    def _analyze_coverage(self, results: List[Dict]) -> Dict[str, int]:
        coverage = {
            "technical_domain": 0,
            "personality_traits": 0,
            "cognitive_traits": 0,
            "assessment_type": 0
        }
        
        for item in results:
            tax = item.get("taxonomy", {})
            if tax.get("technical_domain"): coverage["technical_domain"] += 1
            if tax.get("personality_traits"): coverage["personality_traits"] += 1
            if tax.get("cognitive_traits"): coverage["cognitive_traits"] += 1
            if tax.get("assessment_type"): coverage["assessment_type"] += 1
            
        return coverage

    def rerank(self, query: str, candidates: List[Dict[str, Any]], required_skills: Dict[str, List[str]] = None, soft_negatives: List[str] = None, weights: Dict[str, float] = None, top_k: int = 10) -> Dict[str, Any]:
        if not candidates:
            return {"reranked_results": [], "coverage": {}}
            
        required_skills = required_skills or {}
        soft_negatives = [s.lower() for s in (soft_negatives or [])]
        weights = weights or {"technical_domain": 2.0, "assessment_type": 1.0, "soft_skills": 1.0, "cognitive_traits": 1.0}
            
        # Base semantic scores are bypassed to save memory. 
        # We rely on the hybrid retrieval's RRF score as our foundation.
        pass
        
        # Apply Query-Adaptive Dynamic Weighting
        valid_candidates = []
        for i, item in enumerate(candidates):
            # Base score is derived from the retrieval RRF score
            score = item.get('rrf_score', 0.0) * 10.0 
            tax = item.get("taxonomy", {})
            
            # We set a minimum floor so scores don't collapse into infinity
            min_score = -10.0
            
            # Check Soft Negatives (Soft degradation)
            all_candidate_terms = []
            for vals in tax.values():
                if isinstance(vals, list): all_candidate_terms.extend([v.lower() for v in vals])
                elif isinstance(vals, str): all_candidate_terms.append(vals.lower())
                
            for neg in soft_negatives:
                if neg in all_candidate_terms or neg in item['name'].lower():
                    score *= 0.5 # 50% penalty (was 20%)
                    
            # Catastrophic Lexical Trap Fix (e.g. Java vs JavaScript)
            req_tech = [t.lower() for t in required_skills.get("technical_domain", [])]
            cand_tech = [t.lower() for t in tax.get("technical_domain", [])]
            
            if "java" in req_tech and "javascript" in cand_tech and "java" not in cand_tech:
                # If they explicitly asked for Java, and candidate is JavaScript BUT NOT Java
                score -= 5.0 # Massive penalty
                
            if "c#" in req_tech and "c++" in cand_tech and "c#" not in cand_tech:
                score -= 5.0
                    
            # Check Required Constraints with Dynamic Weights
            for domain, req_list in required_skills.items():
                if not req_list: continue
                domain_weight = weights.get(domain, 1.0)
                
                candidate_domain_terms = tax.get(domain, [])
                if isinstance(candidate_domain_terms, str): candidate_domain_terms = [candidate_domain_terms]
                candidate_domain_terms_lower = [t.lower() for t in candidate_domain_terms]
                
                for req in req_list:
                    req_lower = req.lower()
                    if any(req_lower in ct for ct in candidate_domain_terms_lower) or req_lower in item['name'].lower():
                        score += (0.1 * domain_weight) # Boost
                    else:
                        score -= (0.5 * domain_weight) # Penalize strongly on high weight mismatch (was 0.2)
            
            # Bounded penalties to prevent ranking collapse
            item['rerank_score'] = max(min_score, score)
            valid_candidates.append(item)
            
        reranked = sorted(valid_candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        coverage = self._analyze_coverage(reranked)
        
        return {
            "reranked_results": reranked,
            "coverage": coverage
        }

if __name__ == "__main__":
    from src.engine.retrieval.hybrid_retriever import HybridRetriever
    
    # Simple test
    hybrid = HybridRetriever(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    reranker = Reranker()
    
    query = "software engineering python"
    print(f"Executing Hybrid Retrieval for: '{query}'")
    hybrid_results = hybrid.search(query, top_k=20)
    
    print("\nExecuting Reranking...")
    final_output = reranker.rerank(query, hybrid_results, top_k=5)
    
    print("\nTop 5 Reranked Results:")
    for r in final_output["reranked_results"]:
        print(f"[{r['rerank_score']:.4f}] {r['name']}")
        
    print("\nCoverage Analysis:")
    print(final_output["coverage"])
