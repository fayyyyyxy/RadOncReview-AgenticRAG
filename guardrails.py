"""
Guardrails: Safety and compliance checks for input/output.
Handles PHI filtering, content validation, and version consistency.
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from config import GuardrailConfig


class GuardrailViolation(Enum):
    """Types of guardrail violations."""
    PHI_DETECTED = "phi_detected"
    INPUT_TOO_LONG = "input_too_long"
    BLOCKED_CONTENT = "blocked_content"
    UNSAFE_RECOMMENDATION = "unsafe_recommendation"
    VERSION_MISMATCH = "version_mismatch"


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    violations: List[GuardrailViolation]
    filtered_text: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputGuardrail:
    """Validates and filters user input."""
    
    # PHI patterns
    PHI_PATTERNS = [
        # SSN
        (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', 'SSN'),
        # Phone numbers
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', 'Phone'),
        # Email
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
        # Dates that might be DOB (with context)
        (r'\b(DOB|date of birth|born on)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DOB'),
        # Medical record numbers
        (r'\b(MRN|medical record|patient id)[:\s#]*[A-Z0-9]{5,}\b', 'MRN'),
        # Names with context
        (r'\b(patient name|patient:|name:)[:\s]*[A-Z][a-z]+ [A-Z][a-z]+\b', 'Name'),
    ]
    
    # Potentially harmful query patterns
    HARMFUL_PATTERNS = [
        (r'\b(lethal|fatal) dose\b', 'Potentially harmful query about lethal doses'),
        (r'\bhow to (harm|hurt|kill)\b', 'Potentially harmful intent'),
        (r'\bbypass safety\b', 'Attempt to bypass safety measures'),
    ]
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
    
    def check(self, text: str) -> GuardrailResult:
        """Check input text against all guardrails."""
        violations = []
        warnings = []
        filtered_text = text
        
        # Check input length
        if len(text) > self.config.max_input_length:
            violations.append(GuardrailViolation.INPUT_TOO_LONG)
            warnings.append(f"Input exceeds maximum length of {self.config.max_input_length} characters")
        
        # Check for PHI if enabled
        if self.config.enable_phi_filter:
            phi_found, filtered_text, phi_warnings = self._filter_phi(filtered_text)
            if phi_found:
                violations.append(GuardrailViolation.PHI_DETECTED)
                warnings.extend(phi_warnings)
        
        # Check for blocked terms
        for term in self.config.blocked_terms:
            if term.lower() in text.lower():
                violations.append(GuardrailViolation.BLOCKED_CONTENT)
                warnings.append(f"Blocked term detected: {term}")
        
        # Check for harmful patterns
        for pattern, warning in self.HARMFUL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(GuardrailViolation.BLOCKED_CONTENT)
                warnings.append(warning)
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            filtered_text=filtered_text,
            warnings=warnings
        )
    
    def _filter_phi(self, text: str) -> Tuple[bool, str, List[str]]:
        """Filter PHI from text."""
        phi_found = False
        warnings = []
        filtered = text
        
        for pattern, phi_type in self.PHI_PATTERNS:
            matches = re.findall(pattern, filtered, re.IGNORECASE)
            if matches:
                phi_found = True
                warnings.append(f"Potential {phi_type} detected and filtered")
                # Replace with placeholder
                filtered = re.sub(pattern, f'[{phi_type}_REDACTED]', filtered, flags=re.IGNORECASE)
        
        return phi_found, filtered, warnings


class OutputGuardrail:
    """Validates generated output for safety and compliance."""
    
    # Unsafe medical recommendation patterns
    UNSAFE_PATTERNS = [
        (r'\bdo not (consult|see|visit) (a |your )?(doctor|physician|oncologist)\b', 
         'Discouraging medical consultation'),
        (r'\bstop (taking|your) (medication|treatment)\b',
         'Advising to stop treatment without medical supervision'),
        (r'\b(guarantee|definitely|certainly) (cure|work)\b',
         'Making absolute medical guarantees'),
        (r'\binstead of (chemotherapy|surgery|radiation)\b.*\buse\b',
         'Suggesting alternatives to standard treatment'),
    ]
    
    # Required disclaimer patterns for certain content
    DISCLAIMER_TRIGGERS = [
        (r'\bdose\b.*\b(gy|cgy)\b', 'dose_disclaimer'),
        (r'\b(recommend|suggest|advise)\b.*\btreatment\b', 'treatment_disclaimer'),
    ]
    
    DISCLAIMERS = {
        'dose_disclaimer': "Note: Dose recommendations should be verified with current guidelines and adapted to individual patient factors.",
        'treatment_disclaimer': "This information is for educational purposes. Treatment decisions should be made in consultation with the patient's care team."
    }
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
    
    def check(self, text: str) -> GuardrailResult:
        """Check output text for safety issues."""
        violations = []
        warnings = []
        
        # Check for unsafe patterns
        for pattern, warning in self.UNSAFE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(GuardrailViolation.UNSAFE_RECOMMENDATION)
                warnings.append(warning)
        
        # Check if disclaimers are needed
        needed_disclaimers = []
        for pattern, disclaimer_type in self.DISCLAIMER_TRIGGERS:
            if re.search(pattern, text, re.IGNORECASE):
                needed_disclaimers.append(disclaimer_type)
        
        if needed_disclaimers:
            warnings.append(f"Consider adding disclaimers: {', '.join(needed_disclaimers)}")
        
        return GuardrailResult(
            passed=len(violations) == 0,
            violations=violations,
            filtered_text=text,
            warnings=warnings
        )
    
    def add_disclaimers(self, text: str) -> str:
        """Add appropriate disclaimers to the response."""
        disclaimers_to_add = set()
        
        for pattern, disclaimer_type in self.DISCLAIMER_TRIGGERS:
            if re.search(pattern, text, re.IGNORECASE):
                disclaimers_to_add.add(disclaimer_type)
        
        if disclaimers_to_add:
            disclaimer_text = "\n\n---\n" + "\n".join([
                self.DISCLAIMERS[d] for d in disclaimers_to_add
            ])
            return text + disclaimer_text
        
        return text


class VersionGuardrail:
    """Ensures guideline version consistency."""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        # Track guideline versions used in current session
        self.versions_used = {}
    
    def check_version_consistency(
        self, 
        source_path: str, 
        version: Optional[str] = None
    ) -> GuardrailResult:
        """Check if retrieved content is from consistent guideline versions."""
        if not self.config.enable_version_check:
            return GuardrailResult(passed=True, violations=[])
        
        # Extract guideline name from path
        parts = source_path.split(' > ')
        guideline_name = parts[0] if parts else "Unknown"
        
        if guideline_name in self.versions_used:
            if version and self.versions_used[guideline_name] != version:
                return GuardrailResult(
                    passed=False,
                    violations=[GuardrailViolation.VERSION_MISMATCH],
                    warnings=[f"Version mismatch for {guideline_name}: "
                             f"using {self.versions_used[guideline_name]} and {version}"]
                )
        else:
            self.versions_used[guideline_name] = version or "latest"
        
        return GuardrailResult(passed=True, violations=[])
    
    def reset_session(self):
        """Reset version tracking for new session."""
        self.versions_used = {}


class GuardrailManager:
    """Manages all guardrails."""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)
        self.version_guardrail = VersionGuardrail(config)
    
    def check_input(self, text: str) -> GuardrailResult:
        """Run all input guardrails."""
        return self.input_guardrail.check(text)
    
    def check_output(self, text: str) -> GuardrailResult:
        """Run all output guardrails."""
        return self.output_guardrail.check(text)
    
    def process_output(self, text: str) -> Tuple[str, GuardrailResult]:
        """Check output and add disclaimers if needed."""
        result = self.check_output(text)
        
        if result.passed:
            processed_text = self.output_guardrail.add_disclaimers(text)
            return processed_text, result
        
        return text, result
    
    def reset(self):
        """Reset all stateful guardrails."""
        self.version_guardrail.reset_session()
