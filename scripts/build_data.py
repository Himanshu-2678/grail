import json
import re
from pathlib import Path
from typing import Dict, Any, List

def extract_seniority(job_levels: List[str]) -> List[str]:
    mapping = {
        "Entry-Level": "Junior",
        "Graduate": "Junior",
        "Mid-Professional": "Mid",
        "Professional Individual Contributor": "Mid",
        "Supervisor": "Manager",
        "Front Line Manager": "Manager",
        "Manager": "Manager",
        "Director": "Senior",
        "Executive": "Senior",
        "General Population": "All"
    }
    seniorities = set()
    for level in job_levels:
        if level in mapping:
            seniorities.add(mapping[level])
    return list(seniorities)

def normalize_catalog(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f, strict=False)

    normalized = []
    
    for item in data:
        name = item.get("name", "")
        desc = item.get("description", "")
        keys = item.get("keys", [])
        job_levels = item.get("job_levels", [])
        
        # Determine assessment type
        assessment_type = "Other"
        if "Simulations" in keys:
            assessment_type = "Simulation"
        elif "Knowledge & Skills" in keys:
            assessment_type = "Technical Skills"
        elif "Ability & Aptitude" in keys:
            assessment_type = "Cognitive Ability"
        elif "Personality & Behavior" in keys:
            assessment_type = "Personality"
        elif "Competencies" in keys:
            assessment_type = "Competency"
            
        # Determine technical domain (simple heuristic from name for now)
        technical_domain = []
        if assessment_type == "Technical Skills":
            if ".NET" in name or "C#" in name:
                technical_domain.append(".NET")
            if "Java" in name and "JavaScript" not in name:
                technical_domain.append("Java")
            if "JavaScript" in name:
                technical_domain.append("JavaScript")
            if "Adobe" in name:
                technical_domain.append("Design/Adobe")
            if "Accounting" in name or "Accounts" in name:
                technical_domain.append("Accounting")
            if "Engineering" in name:
                technical_domain.append("Engineering")
            if "Angular" in name:
                technical_domain.append("Frontend/Angular")
            if "Apache" in name or "Hadoop" in name or "Spark" in name:
                technical_domain.append("Data Engineering/Big Data")
            if "AWS" in name:
                technical_domain.append("Cloud/AWS")
            if "Android" in name:
                technical_domain.append("Mobile/Android")
                
        # Basic trait extraction from description (could be enhanced later)
        soft_skills = []
        cognitive_traits = []
        personality_traits = []
        
        desc_lower = desc.lower()
        if "communication" in desc_lower: soft_skills.append("communication")
        if "leadership" in desc_lower: soft_skills.append("leadership")
        if "stakeholder" in desc_lower: soft_skills.append("stakeholder management")
        
        if "numerical" in desc_lower: cognitive_traits.append("numerical reasoning")
        if "verbal" in desc_lower: cognitive_traits.append("verbal reasoning")
        if "logical" in desc_lower: cognitive_traits.append("logical reasoning")
        
        if "resilience" in desc_lower: personality_traits.append("resilience")
        if "teamwork" in desc_lower: personality_traits.append("teamwork")
        
        normalized_item = {
            "id": item.get("entity_id"),
            "name": name,
            "url": item.get("link", ""),
            "description": desc,
            "duration_raw": item.get("duration", ""),
            "is_adaptive": item.get("adaptive") == "yes",
            "is_remote": item.get("remote") == "yes",
            
            # Normalized Taxonomy
            "taxonomy": {
                "seniority": extract_seniority(job_levels),
                "role_alignment": job_levels,
                "assessment_type": assessment_type,
                "technical_domain": technical_domain,
                "soft_skills": soft_skills,
                "cognitive_traits": cognitive_traits,
                "personality_traits": personality_traits
            }
        }
        normalized.append(normalized_item)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=2)
        
    print(f"Normalized {len(normalized)} items to {output_path}")

if __name__ == "__main__":
    input_file = Path(r"c:\Users\himan\Desktop\Projects\grail\data\raw\catalogue.txt")
    output_file = Path(r"c:\Users\himan\Desktop\Projects\grail\data\processed\normalized_catalog.json")
    normalize_catalog(str(input_file), str(output_file))
