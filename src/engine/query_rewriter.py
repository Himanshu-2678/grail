import os
from google import genai
from google.genai import types
from typing import List, Optional
from pydantic import BaseModel, Field

class DomainConstraints(BaseModel):
    technical_domain: List[str] = Field(default_factory=list, description="Technical hard skills like Java, React, Accounting.")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills like communication, leadership.")
    cognitive_traits: List[str] = Field(default_factory=list, description="Cognitive abilities like numerical reasoning.")
    assessment_type: List[str] = Field(default_factory=list, description="Types of assessments.")

class DomainWeights(BaseModel):
    technical_domain: float = Field(default=2.0, description="Weight for technical matches")
    soft_skills: float = Field(default=1.0, description="Weight for soft skills")
    cognitive_traits: float = Field(default=1.0, description="Weight for cognitive traits")
    assessment_type: float = Field(default=1.0, description="Weight for assessment type")

class QueryExpansion(BaseModel):
    expanded_terms: List[str] = Field(description="Taxonomy terms mapping to the user's intent.")
    intent_detected: str = Field(description="Overall intent of the query.")
    required_constraints: DomainConstraints = Field(description="MANDATORY terms the user insists upon.")
    hard_negative_constraints: DomainConstraints = Field(description="HARD EXCLUSIONS the user strictly avoids.")
    soft_negative_constraints: List[str] = Field(description="SOFT preferences for exclusion.")
    constraint_weights: DomainWeights = Field(description="Dynamic weights based on user emphasis.")

class SemanticQueryExpander:
    def __init__(self):
        # Assumes GEMINI_API_KEY is set in environment
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        
        # We define our taxonomy so the LLM knows what to map to
        self.taxonomy = {
            "soft_skills": ["communication", "leadership", "stakeholder management"],
            "cognitive_traits": ["numerical reasoning", "verbal reasoning", "logical reasoning"],
            "personality_traits": ["resilience", "teamwork"]
        }
        
    def _deterministic_extraction(self, query: str) -> dict:
        import re
        hard_negatives = {"technical_domain": [], "soft_skills": [], "cognitive_traits": [], "assessment_type": []}
        
        KNOWN_DOMAINS = {
            "technical_domain": {
                "javascript": ["javascript", "js", "nodejs", "node"],
                "java": ["java", "jdk", "j2ee"],
                "c++": ["c++", "cpp"],
                "c#": ["c#", "csharp", ".net"],
                "python": ["python", "py"],
                "frontend": ["frontend", "front end", "react", "angular", "vue"],
                "backend": ["backend", "back end"],
                "sales": ["sales"]
            },
            "assessment_type": {
                "personality": ["personality", "behavioral", "psychometric"],
                "coding": ["coding", "programming", "automata"]
            }
        }
        
        negation_words = [r"\bnot\b", r"\bexclude\b", r"\bavoid\b", r"\bwithout\b", r"\bexcept\b", r"\banything but\b", r"\bno\b"]
        neg_pattern = "(?:" + "|".join(negation_words) + ")"
        
        query_lower = query.lower()
        
        for domain_type, entities in KNOWN_DOMAINS.items():
            for canonical, aliases in entities.items():
                for alias in aliases:
                    # Look for: negation word followed by 0-3 words, then the alias
                    pattern = rf"{neg_pattern}\s+(?:\w+\s+){{0,3}}\b{re.escape(alias)}\b"
                    if re.search(pattern, query_lower):
                        hard_negatives[domain_type].append(canonical)
                        break 
        
        return hard_negatives

    def expand(self, query: str) -> dict:
        prompt = f"""
        You are an ontology mapping assistant for the SHL assessment catalog.
        Your task is to parse a raw user query into hierarchical, taxonomy-aware constraints.
        
        Taxonomy Domains:
        Soft Skills: {', '.join(self.taxonomy['soft_skills'])}
        Cognitive: {', '.join(self.taxonomy['cognitive_traits'])}
        Personality: {', '.join(self.taxonomy['personality_traits'])}
        
        Rules:
        1. Classify negations properly. 
           HARD NEGATION: "no javascript", "exclude sales" -> mapped to `hard_negative_constraints` dict by taxonomy domain.
           SOFT NEGATION: "less theoretical", "preferably no personality tests" -> mapped to `soft_negative_constraints` list.
        2. Map constraints to canonical taxonomy domains: `technical_domain`, `soft_skills`, `cognitive_traits`, `assessment_type`.
        3. Assign dynamic `constraint_weights`. If a user explicitly asks for "communication", give `soft_skills` a higher weight (e.g. 2.5). Base technical weight is usually 2.0.
        4. STATE MUTATION (CRITICAL): The raw query is a full conversational transcript. If the user contradicts or changes their mind (e.g., "actually", "instead", "change to", "not anymore"), you MUST drop the old constraints. "Need Java... actually change to Python" -> require ONLY Python.
        
        Raw Query: "{query}"
        """
        
        # We use structured output to strictly control the LLM's response
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QueryExpansion,
                ),
            )
            result = response.parsed
        except Exception as e:
            import logging
            logging.warning(f"Gemini API failure during expansion: {e}. Falling back to deterministic extraction.")
            result = None
            
        if not result:
            return {
                "expanded_query": query, 
                "required": {}, 
                "hard_negative": self._deterministic_extraction(query), 
                "soft_negative": [], 
                "weights": {"technical_domain": 2.0, "soft_skills": 1.0, "assessment_type": 1.0}
            }
            
        expanded = f"{query} {' '.join(result.expanded_terms)}"
        
        deterministic_negatives = self._deterministic_extraction(query)
        llm_negatives = result.hard_negative_constraints.model_dump()
        
        # Merge deterministic negatives with LLM negatives (Deterministic overrides)
        merged_negatives = {}
        for k in set(llm_negatives.keys()).union(deterministic_negatives.keys()):
            l_list = llm_negatives.get(k, [])
            d_list = deterministic_negatives.get(k, [])
            merged_negatives[k] = list(set(l_list + d_list))
            
        return {
            "expanded_query": expanded,
            "required": result.required_constraints.model_dump(),
            "hard_negative": merged_negatives,
            "soft_negative": result.soft_negative_constraints,
            "weights": result.constraint_weights.model_dump()
        }

if __name__ == "__main__":
    expander = SemanticQueryExpander()
    q = "good with stakeholders"
    print(f"Original: {q}")
    expanded = expander.expand(q)
    print(f"Expanded: {expanded}")
