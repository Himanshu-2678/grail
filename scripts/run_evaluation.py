import os
import json
from typing import List, Dict
from src.engine.orchestrator import GrailOrchestrator

class EvaluationHarness:
    def __init__(self, data_path: str):
        self.orchestrator = GrailOrchestrator(data_path)
        
        # Adversarial Test Set
        self.test_cases = [
            # 1. Lexical Trap (Testing negative constraints)
            {"query": "java engineer but absolutely not javascript", "expected_concept": "Java", "negative_trap": "JavaScript", "type": "lexical_trap"},
            {"query": "c# developer do not want c++", "expected_concept": "C#", "negative_trap": "C++", "type": "lexical_trap"},
            
            # 2. Semantic Traps / Aliases
            {"query": "good with stakeholders", "expected_concept": "Communication", "type": "semantic"},
            {"query": "handles pressure well", "expected_concept": "Resilience", "type": "semantic"},
            {"query": "crunching numbers", "expected_concept": "Numerical", "type": "semantic"},
            
            # 3. Ambiguous Triggers (Should trigger clarification)
            {"query": "software", "expected_concept": "clarify", "type": "ambiguous"},
            {"query": "i need a test", "expected_concept": "clarify", "type": "ambiguous"},
            {"query": "manager", "expected_concept": "clarify", "type": "ambiguous"},
            
            # 4. Multi-Intent
            {"query": "need coding plus personality assessment for java backend", "expected_concept": ["Java", "Personality"], "type": "multi_intent"},
            {"query": "test numerical reasoning and communication", "expected_concept": ["Numerical", "Communication"], "type": "multi_intent"},
            
            # 5. Direct Domain Matches
            {"query": "python backend", "expected_concept": "Python", "type": "direct"},
            {"query": "accounting software accounts payable", "expected_concept": "Accounts Payable", "type": "direct"},
            {"query": "frontend react", "expected_concept": "React", "type": "direct"},
            
            # 6. False Trap (Coexistence, NOT negation. Should NOT veto)
            {"query": "Java architect with JavaScript exposure", "expected_concept": "Java", "type": "false_trap"},
            
            # 7. Contradictory Refinement (Simulated state mutation)
            {"query": "Need Java backend test. Actually not coding focused anymore", "expected_concept": "Java", "negative_trap": "automata", "type": "contradictory_refinement"},
            
            # 8. Soft Ambiguity
            {"query": "Need assessment for someone analytical", "expected_concept": "clarify", "type": "ambiguous"},
        ]
        
    def evaluate(self) -> Dict:
        results = {
            "total": len(self.test_cases),
            "top_1_hits": 0,
            "top_3_hits": 0,
            "clarifications_triggered": 0,
            "trap_avoidance_rate": 0,
            "details": []
        }
        
        trap_tests = 0
        traps_avoided = 0
        
        import time
        for case in self.test_cases:
            query = case["query"]
            expected = case["expected_concept"]
            case_type = case.get("type")
            
            # Rate limiting for free tier
            time.sleep(5)
            
            print(f"Evaluating Query: '{query}'")
            messages = [{"role": "user", "content": query}]
            
            snapshot = None
            retries = 0
            while retries < 5:
                try:
                    snapshot = self.orchestrator.process_chat(messages)
                    break
                except Exception as e:
                    if "429" in str(e) or "503" in str(e):
                        print(f"  API Error ({e}). Retrying in 60s... (Attempt {retries+1}/5)")
                        time.sleep(60)
                        retries += 1
                    else:
                        print(f"  Error processing query: {e}")
                        break
            
            if not snapshot:
                print(f"  Failed to evaluate query: {query}")
                continue

            top_candidates = snapshot["recommendations"]
            clarification = len(top_candidates) == 0
            
            # Check clarification
            if clarification:
                results["clarifications_triggered"] += 1
                if expected == "clarify":
                    results["top_1_hits"] += 1
                    results["top_3_hits"] += 1
                    continue
                    
            if expected == "clarify" and not clarification:
                results["details"].append({
                    "query": query, 
                    "status": "FAILED - Did not clarify",
                    "type": case_type,
                    "clarification_triggered": False
                })
                continue
                
            # Check hits
            top_1 = False
            top_3 = False
            trap_avoided = True
            
            if top_candidates:
                top_1_name = top_candidates[0]["name"].lower()
                
                # Check for negative traps
                if "negative_trap" in case:
                    trap_tests += 1
                    for cand in top_candidates[:3]:
                        if case["negative_trap"].lower() in cand["name"].lower():
                            trap_avoided = False
                    if trap_avoided:
                        traps_avoided += 1

                # Match logic for multi-intent
                if isinstance(expected, list):
                    matched_any_top1 = any(e.lower() in top_1_name for e in expected)
                    matched_any_top3 = any(any(e.lower() in c["name"].lower() for e in expected) for c in top_candidates[:3])
                    top_1 = matched_any_top1
                    top_3 = matched_any_top3
                else:
                    exp = expected.lower()
                    top_1 = exp in top_1_name
                    top_3 = any(exp in c["name"].lower() for c in top_candidates[:3])
                    
            if top_1: results["top_1_hits"] += 1
            if top_3: results["top_3_hits"] += 1
                
            results["details"].append({
                "query": query,
                "type": case_type,
                "top_1": top_1,
                "top_3": top_3,
                "top_1_candidate": top_candidates[0]["name"] if top_candidates else "None",
                "clarification_triggered": clarification
            })
            
        results["top_1_accuracy"] = results["top_1_hits"] / results["total"]
        results["top_3_accuracy"] = results["top_3_hits"] / results["total"]
        
        if trap_tests > 0:
            results["trap_avoidance_rate"] = traps_avoided / trap_tests
            
        # Constraint Violation Rate = (Failed traps) / (Total traps)
        results["constraint_violation_rate"] = 1.0 - results.get("trap_avoidance_rate", 1.0)
        
        # Calibration Metrics
        # True Positives: Triggered when ambiguous
        # False Positives: Triggered when clear
        # True Negatives: Didn't trigger when clear
        # False Negatives: Didn't trigger when ambiguous
        clarify_cases = [c for c in self.test_cases if c["expected_concept"] == "clarify"]
        non_clarify_cases = [c for c in self.test_cases if c["expected_concept"] != "clarify"]
        
        tp = sum(1 for c in results["details"] if c.get("type") == "ambiguous" and c["clarification_triggered"])
        fp = sum(1 for c in results["details"] if c.get("type") != "ambiguous" and c["clarification_triggered"])
        fn = sum(1 for c in results["details"] if c.get("type") == "ambiguous" and not c["clarification_triggered"])
        
        results["precision_of_abstention"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results["false_clarification_rate"] = fp / len(non_clarify_cases) if non_clarify_cases else 0.0
            
        return results

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'normalized_catalog.json')
    harness = EvaluationHarness(data_path)
    
    print("\n--- Starting Adversarial Evaluation Harness ---")
    metrics = harness.evaluate()
    
    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nFinal Top-1 Accuracy: {metrics['top_1_accuracy']*100:.1f}%")
    print(f"Final Top-3 Accuracy: {metrics['top_3_accuracy']*100:.1f}%")
    print(f"Trap Avoidance Rate: {metrics.get('trap_avoidance_rate', 0)*100:.1f}%")
    print(f"Constraint Violation Rate: {metrics['constraint_violation_rate']*100:.1f}%")
    print("\n--- Calibration Metrics ---")
    print(f"Precision of Abstention: {metrics['precision_of_abstention']*100:.1f}%")
    print(f"False Clarification Rate: {metrics['false_clarification_rate']*100:.1f}%")
