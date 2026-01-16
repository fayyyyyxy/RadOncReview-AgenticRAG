"""
Query Analyzer: Parses and classifies user queries.
Identifies intent, entities, disease categories, and treatment modalities.
"""
import re
from typing import Optional, List, Tuple

from models import QueryAnalysis, QueryIntent
from llm_client import BaseLLMClient
from config import AgentConfig


class QueryAnalyzer:
    """Analyzes and classifies radiation oncology queries."""
    
    # Intent keyword patterns
    INTENT_PATTERNS = {
        QueryIntent.DOSE_FRACTIONATION: [
            r'\bdose\b', r'\bgy\b', r'\bgray\b', r'\bfraction', r'\bfx\b',
            r'\bhypofractionat', r'\bconventional\s+fraction', r'\bBED\b', r'\bEQD2\b'
        ],
        QueryIntent.TREATMENT_PROTOCOL: [
            r'\bprotocol\b', r'\bguideline\b', r'\btreatment\s+plan', 
            r'\bstandard\s+of\s+care', r'\bregimen\b', r'\bapproach\b'
        ],
        QueryIntent.SIDE_EFFECTS: [
            r'\bside\s+effect', r'\btoxicit', r'\badverse', r'\bcomplication',
            r'\brisk\b', r'\bpneumonitis\b', r'\besophagitis\b', r'\bdermatitis\b'
        ],
        QueryIntent.STAGING: [
            r'\bstag(e|ing)\b', r'\bTNM\b', r'\bAJCC\b', r'\bT[0-4]', r'\bN[0-3]', r'\bM[0-1]'
        ],
        QueryIntent.INDICATIONS: [
            r'\bindication', r'\bwhen\s+to\s+(use|treat)', r'\bcandidate',
            r'\beligib', r'\bappropriate\s+for'
        ],
        QueryIntent.CONTRAINDICATIONS: [
            r'\bcontraindication', r'\bavoid\b', r'\bnot\s+recommend',
            r'\bprecaution', r'\bwhen\s+not\s+to'
        ],
        QueryIntent.COMPARISON: [
            r'\bvs\.?\b', r'\bversus\b', r'\bcompare', r'\bdifference\s+between',
            r'\bbetter\b', r'\bprefer'
        ],
        QueryIntent.DEFINITION: [
            r'\bwhat\s+is\b', r'\bdefine\b', r'\bmeaning\s+of', r'\bexplain\b'
        ]
    }
    
    # Disease category patterns
    DISEASE_PATTERNS = {
        "Breast": [r'\bbreast\b', r'\bmammary\b', r'\bDCIS\b', r'\bIDC\b', r'\bILC\b'],
        "Thorax": [r'\blung\b', r'\bNSCLC\b', r'\bSCLC\b', r'\bthora', r'\bmesothelioma\b', r'\bthymoma\b'],
        "Hematologic": [r'\bleukemia\b', r'\blymphoma\b', r'\bmyeloma\b', r'\bhodgkin', r'\bTBI\b'],
        "CNS": [r'\bbrain\b', r'\bglioma\b', r'\bmeningioma\b', r'\bGBM\b', r'\bcranial\b', r'\bspinal\b'],
        "GI": [r'\besophag', r'\bgastric\b', r'\bpancrea', r'\bliver\b', r'\brectal\b', r'\banal\b', r'\bcolon\b'],
        "GU": [r'\bprostate\b', r'\bbladder\b', r'\brenal\b', r'\bkidney\b', r'\btesticular\b'],
        "Gynecologic": [r'\bcervic', r'\buterine\b', r'\bendometri', r'\bovarian\b', r'\bvulvar\b', r'\bvaginal\b'],
        "Head_Neck": [r'\bhead\s*(and|&)\s*neck\b', r'\bH&?N\b', r'\blaryn', r'\bpharyn', r'\boral\b', r'\bnasopharyn'],
        "Pediatric": [r'\bpediatric\b', r'\bchild', r'\bneuroblastoma\b', r'\bwilms\b', r'\brhabdomyosarcoma\b'],
        "Sarcoma": [r'\bsarcoma\b', r'\bsoft\s+tissue\b', r'\bbone\s+(cancer|tumor)\b'],
        "Skin": [r'\bmelanoma\b', r'\bskin\b', r'\bsquamous\s+cell\b', r'\bbasal\s+cell\b', r'\bmerkel\b']
    }
    
    # Treatment modality patterns
    MODALITY_PATTERNS = {
        "EBRT": [r'\bEBRT\b', r'\bexternal\s+beam\b'],
        "IMRT": [r'\bIMRT\b', r'\bintensity[\s-]modulated\b'],
        "VMAT": [r'\bVMAT\b', r'\bvolumetric[\s-]modulated\b', r'\bRapidArc\b'],
        "SRS": [r'\bSRS\b', r'\bstereotactic\s+radiosurgery\b', r'\bgamma\s*knife\b', r'\bcyberknife\b'],
        "SBRT": [r'\bSBRT\b', r'\bSABR\b', r'\bstereotactic\s+body\b'],
        "Brachytherapy": [r'\bbrachy', r'\bHDR\b', r'\bLDR\b', r'\bimplant\b', r'\bseed\b'],
        "Proton": [r'\bproton\b', r'\bparticle\s+therapy\b', r'\bPBT\b'],
        "Electron": [r'\belectron\b', r'\be[\s-]?beam\b'],
        "TBI": [r'\bTBI\b', r'\btotal\s+body\s+irradiation\b'],
        "TSEI": [r'\bTSEI\b', r'\btotal\s+skin\s+electron\b']
    }
    
    def __init__(self, llm_client: BaseLLMClient, config: AgentConfig):
        self.llm = llm_client
        self.config = config
    
    def _match_patterns(self, text: str, patterns: dict) -> Optional[str]:
        """Match text against a dictionary of regex patterns."""
        text_lower = text.lower()
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category
        return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key medical entities from the query."""
        entities = []
        
        # Extract dose values (e.g., "50 Gy", "2 Gy/fx")
        dose_pattern = r'\b\d+\.?\d*\s*(Gy|cGy|Gy/fx|fx)\b'
        entities.extend(re.findall(dose_pattern, text, re.IGNORECASE))
        
        # Extract staging (e.g., "T2N1M0", "Stage IIIA")
        stage_pattern = r'\b(T[0-4][a-c]?N[0-3][a-c]?M[0-1]|Stage\s+[IVX]+[A-C]?)\b'
        entities.extend(re.findall(stage_pattern, text, re.IGNORECASE))
        
        # Extract anatomical structures
        anatomy_pattern = r'\b(lymph\s+node|chest\s+wall|supraclavicular|axilla|internal\s+mammary|mediastin)\b'
        entities.extend(re.findall(anatomy_pattern, text, re.IGNORECASE))
        
        return list(set(entities))
    
    def _detect_complexity(self, text: str, intent: QueryIntent) -> bool:
        """Determine if the query requires multi-step reasoning."""
        # Multi-part questions
        if ' and ' in text.lower() and '?' in text:
            return True
        
        # Comparison queries often need multiple retrievals
        if intent == QueryIntent.COMPARISON:
            return True
        
        # Questions with multiple conditions
        condition_words = ['if', 'when', 'given', 'assuming', 'in case of']
        if any(word in text.lower() for word in condition_words):
            return True
        
        # Long queries tend to be more complex
        if len(text.split()) > 30:
            return True
        
        return False
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Perform rule-based query analysis."""
        # Detect intent
        intent = QueryIntent.GENERAL
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    intent = intent_type
                    break
            if intent != QueryIntent.GENERAL:
                break
        
        # Detect disease category
        disease_category = self._match_patterns(query, self.DISEASE_PATTERNS)
        
        # Detect treatment modality
        treatment_modality = self._match_patterns(query, self.MODALITY_PATTERNS)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Detect complexity
        is_complex = self._detect_complexity(query, intent)
        
        # Rule-based confidence
        confidence = 0.7
        if disease_category:
            confidence += 0.1
        if treatment_modality:
            confidence += 0.1
        if entities:
            confidence += 0.05
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            disease_category=disease_category,
            treatment_modality=treatment_modality,
            key_entities=entities,
            is_complex=is_complex,
            confidence=min(confidence, 1.0)
        )
    
    def analyze_with_llm(self, query: str) -> QueryAnalysis:
        """Use LLM for more sophisticated query analysis."""
        # First do rule-based analysis
        rule_based = self.analyze(query)
        
        # Use LLM to enhance understanding
        system_prompt = """You are a radiation oncology clinical query analyzer. 
Analyze the user's question and extract structured information.
Be precise and use standard medical terminology."""
        
        analysis_prompt = f"""Analyze this radiation oncology query:
"{query}"

Extract the following information in JSON format:
{{
    "intent": "<one of: treatment_protocol, dose_fractionation, side_effects, staging, indications, contraindications, comparison, definition, general>",
    "disease_category": "<one of: Breast, Thorax, Hematologic, CNS, GI, GU, Gynecologic, Head_Neck, Pediatric, Sarcoma, Skin, or null>",
    "treatment_modality": "<one of: EBRT, IMRT, VMAT, SRS, SBRT, Brachytherapy, Proton, Electron, TBI, TSEI, or null>",
    "key_entities": ["<list of key medical terms, doses, stages, anatomical structures>"],
    "temporal_context": "<e.g., adjuvant, neoadjuvant, definitive, palliative, postoperative, or null>",
    "is_complex": <true if requires multi-step reasoning, false otherwise>,
    "reasoning": "<brief explanation of your analysis>"
}}"""
        
        try:
            result = self.llm.generate_structured(
                analysis_prompt,
                output_schema={
                    "intent": "string",
                    "disease_category": "string or null",
                    "treatment_modality": "string or null", 
                    "key_entities": "array of strings",
                    "temporal_context": "string or null",
                    "is_complex": "boolean",
                    "reasoning": "string"
                },
                system_prompt=system_prompt
            )
            
            if "error" not in result:
                # Map string intent to enum
                intent_map = {
                    "treatment_protocol": QueryIntent.TREATMENT_PROTOCOL,
                    "dose_fractionation": QueryIntent.DOSE_FRACTIONATION,
                    "side_effects": QueryIntent.SIDE_EFFECTS,
                    "staging": QueryIntent.STAGING,
                    "indications": QueryIntent.INDICATIONS,
                    "contraindications": QueryIntent.CONTRAINDICATIONS,
                    "comparison": QueryIntent.COMPARISON,
                    "definition": QueryIntent.DEFINITION,
                    "general": QueryIntent.GENERAL
                }
                
                return QueryAnalysis(
                    original_query=query,
                    intent=intent_map.get(result.get("intent", "general"), QueryIntent.GENERAL),
                    disease_category=result.get("disease_category"),
                    treatment_modality=result.get("treatment_modality"),
                    key_entities=result.get("key_entities", []),
                    temporal_context=result.get("temporal_context"),
                    is_complex=result.get("is_complex", False),
                    confidence=0.9  # LLM analysis has higher confidence
                )
        except Exception as e:
            print(f"LLM analysis failed, falling back to rule-based: {e}")
        
        return rule_based
