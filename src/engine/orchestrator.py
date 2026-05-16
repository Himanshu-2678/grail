from typing import Dict, Any, List
from src.engine.retrieval.hybrid_retriever import HybridRetriever
from src.engine.retrieval.reranker import Reranker
from src.engine.query_rewriter import SemanticQueryExpander

class GrailOrchestrator:
    def __init__(self, data_path: str):
        self.retriever = HybridRetriever(data_path)
        self.reranker = Reranker()
        self.expander = SemanticQueryExpander()
        
    def _evaluate_calibration_policy(self, query: str, reranked_results: List[Dict], coverage: Dict) -> Dict[str, Any]:
        """
        Entropy-Based Calibration & Structural Ambiguity Gating:
        Clarify when the query lacks semantic specificity, OR when 
        candidate_domains_are_mutually_unrelated AND query_specificity_low.
        """
        import re
        
        # 0. Minimum Specificity Gating (Structural Ambiguity)
        query_words = set(re.findall(r'\b[a-zA-Z]+\b', query.lower()))
        low_specificity_terms = {
            "developer", "engineer", "manager", "analyst", "assessment", "test", "role", "job", "position",
            "need", "an", "for", "a", "i", "want", "some", "looking", "hire", "hiring", "the", "me", "show", "give", "find"
        }
        
        if query_words and query_words.issubset(low_specificity_terms):
            return {
                "trigger_clarification": True, 
                "reason": "Query lacks structural specificity",
                "metrics": {}
            }

        if len(reranked_results) < 2:
            return {"trigger_clarification": False, "reason": "Not enough candidates", "metrics": {}}
            
        # 1. Low Specificity
        active_constraints = sum(1 for v in coverage.values() if v > 0)
        query_specificity_low = len(query.split()) < 3 or active_constraints == 0
        
        # 2. Candidate Diversity (Entropy) - Semantic incompatibility
        # Check technical domains in the top 5 candidates. 
        # If there are multiple unrelated technical domains (e.g. 3+), the query is highly ambiguous.
        top_5_tech_domains = set()
        for r in reranked_results[:5]:
            tech = r.get("taxonomy", {}).get("technical_domain", [])
            if isinstance(tech, str): tech = [tech]
            top_5_tech_domains.update([t.lower() for t in tech])
            
        candidate_domains_mutually_unrelated = len(top_5_tech_domains) >= 3
        
        trigger = query_specificity_low and candidate_domains_mutually_unrelated
        
        return {
            "trigger_clarification": trigger,
            "metrics": {
                "query_specificity_words": len(query.split()),
                "active_constraints": active_constraints,
                "distinct_technical_domains_in_top_5": len(top_5_tech_domains),
                "top_5_domains": list(top_5_tech_domains)
            }
        }

    def _generate_reply(self, query: str, candidates: List[Dict], trigger_clarification: bool) -> str:
        if trigger_clarification:
            return "I need a bit more detail to give you the best recommendation. Could you clarify what specific skills or assessment types you are looking for?"
            
        if not candidates:
            return "I couldn't find any assessments matching all your strict requirements. Could you try broadening your search?"
            
        names = [c["name"] for c in candidates[:3]]
        return f"Based on your constraints, I've found some highly relevant assessments. I'd recommend starting with {names[0]}."

    def process_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # 0. Reconstruct state
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        query = " | ".join(user_messages) if user_messages else ""
        
        try:
            expansion_data = self.expander.expand(query)
            expanded_query = expansion_data["expanded_query"]
            required_constraints = expansion_data["required"]
            hard_negatives = expansion_data["hard_negative"]
            soft_negatives = expansion_data["soft_negative"]
            weights = expansion_data["weights"]
        except Exception as e:
            print(f"Expander failed: {e}")
            raise e
            
        # 1. Hybrid Retrieval
        hybrid_results = self.retriever.search(expanded_query, top_k=100)
        
        # 1.5 Lexical Veto Layer
        import re
        clean_results = []
        for res in hybrid_results:
            tax = res.get("taxonomy", {})
            vetoed = False
            for domain, neg_list in hard_negatives.items():
                if not neg_list: continue
                
                # We check the entire textual footprint of the candidate
                candidate_domain_terms = tax.get(domain, [])
                if isinstance(candidate_domain_terms, str):
                    candidate_domain_terms = [candidate_domain_terms]
                candidate_domain_terms_lower = [t.lower() for t in candidate_domain_terms]
                
                # Also gather full text context
                full_text = f"{res.get('name', '')} {res.get('description', '')}".lower()
                
                for neg_term in neg_list:
                    neg_lower = neg_term.lower()
                    
                    # Exact taxonomy match
                    if any(neg_lower == ct for ct in candidate_domain_terms_lower):
                        vetoed = True
                        break
                        
                    # Word boundary match in full text (e.g. "java" won't match "javascript")
                    # We escape neg_lower to prevent regex injection errors
                    pattern = r'\b' + re.escape(neg_lower) + r'\b'
                    if re.search(pattern, full_text):
                        vetoed = True
                        break
                        
                if vetoed: break
            if not vetoed:
                clean_results.append(res)
                
        # 2. Reranking
        reranked_output = self.reranker.rerank(
            expanded_query, 
            clean_results[:20], 
            required_skills=required_constraints,
            soft_negatives=soft_negatives,
            weights=weights,
            top_k=10
        )
        
        reranked = reranked_output["reranked_results"]
        coverage = reranked_output["coverage"]
        
        # 3. Calibration
        calibration = self._evaluate_calibration_policy(query, reranked, coverage)
        trigger_clarification = calibration["trigger_clarification"]
        
        # 4. Construct Response
        reply = self._generate_reply(query, reranked, trigger_clarification)
        
        # Determine end of conversation (if user accepts the recommendations)
        latest_msg = user_messages[-1].lower() if user_messages else ""
        closure_phrases = ["thanks", "thank you", "perfect", "that works", "looks good", "great"]
        end_of_conv = any(p in latest_msg for p in closure_phrases) and not trigger_clarification
        
        # If clarifying or off-topic, we MUST return []
        recommendations = []
        if not trigger_clarification:
            for r in reranked:
                tax = r.get("taxonomy", {})
                a_type = tax.get("assessment_type", "Other")
                
                # Map to evaluator codes (K = Knowledge, P = Personality, A = Ability)
                type_map = {
                    "Technical Skills": "K",
                    "Personality": "P",
                    "Cognitive Ability": "A",
                    "Simulation": "S",
                    "Competency": "C",
                    "Other": "O"
                }
                test_type_code = type_map.get(a_type, "K") # default to K if unknown
                
                recommendations.append({
                    "name": r["name"],
                    "url": r.get("url", "https://www.shl.com/products/product-catalog/"),
                    "test_type": test_type_code
                })
        
        return {
            "reply": reply,
            "recommendations": recommendations,
            "end_of_conversation": end_of_conv
        }
