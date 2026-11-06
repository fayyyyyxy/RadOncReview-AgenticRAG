"""
Critic edits: apply the sentence edits a critique returns (docs/rag_improvement_plan_v2.md, E4).
Code makes the change, so every sentence the critique did not name stays exactly as written; a rewrite re-sampled
the whole answer, which was the run-to-run noise, and grew it from about 2,600 to 4,600 characters.
"""
import re
from typing import Dict, List, Tuple

from .citation_check import GROUP

# Citation groups (with the spaces before them) and bold or italic marks: the critique may copy a sentence with or
# without them, or write "[5] [6]" where the answer has "[5], [6]"
IGNORED = re.compile(rf"{GROUP.pattern}|\*+")
SAME = str.maketrans({"–": "-", "—": "-", "‑": "-", "“": '"', "”": '"', "‘": "'", "’": "'"})
ACTIONS = ("replace", "insert_after")
# Gaps searched for per answer, and passages kept per gap: the old re-plan appended a median of 28 passages
GAP_LIMIT = 3


def valid_edits(edits) -> List[Dict]:
    """The edits code can apply: a known action, a sentence to find and text to put in."""
    return [e for e in edits or [] if isinstance(e, dict) and e.get("action") in ACTIONS
            and str(e.get("sentence", "")).strip() and str(e.get("text", "")).strip()]


class GapFiller:
    """Writes a sentence for a gap the critique verified, from the passages a search for it found, or nothing."""

    SYSTEM_PROMPT = "You add to a radiation oncology answer only what the given passages state."

    def __init__(self, llm_client):
        self.llm = llm_client

    def fill(self, query: str, answer: str, gaps: List[Dict], context: str) -> List[Dict]:
        """Edits that put each gap's sentence in the answer; none for a gap no new passage covers."""
        items = "\n".join(f"- {g['item']}" for g in gaps)
        # ⚠️ A passage can state the number for another disease: SAR-016's "CTV = GTV + 1.0 cm" came from a
        # pediatric Ewing protocol, which the citation check would pass for an adult soft-tissue sarcoma
        prompt = f"""The answer below to a clinical question lacks the items listed, because the passages it was \
written from do not cover them. A search for each item found the new passages below.

Question: {query}

Answer:
{answer}

Items:
{items}

New passages:
{context}

For each item, write one sentence only if a new passage states it for this patient's condition and setting: the \
same disease, the same setting (for example definitive, preoperative or postoperative) and the same prior radiation \
status. Cite that passage by its number, and state nothing the passage does not state. If no new passage states an \
item for this patient, leave the item out.
Each edit copies exactly one sentence of the answer as "sentence": the sentence that says the item is missing, which \
the new sentence replaces ("action": "replace"), or, if there is none, the sentence the new one should follow \
("action": "insert_after").

Respond in JSON format:
{{"edits": [{{"kind": "fill", "action": "<replace or insert_after>", "sentence": "<copied exactly from the answer>", \
"text": "<the new sentence with its citation>"}}]}}"""
        try:
            reply = self.llm.generate_structured(prompt, output_schema={"edits": "array"},
                                                 system_prompt=self.SYSTEM_PROMPT)
        except Exception as e:
            print(f"Gap fill failed: {e}")
            return []
        # ⚠️ Fail closed: an unusable reply adds nothing, so the stated gap stays
        return [] if "error" in reply else valid_edits(reply.get("edits"))


def _normalized(text: str) -> Tuple[str, List[int]]:
    """The text without citations and emphasis, with spaces collapsed and case folded, and the index in the
    original of each character kept."""
    ignored = {i for m in IGNORED.finditer(text) for i in range(m.start(), m.end())}
    chars, where = [], []
    for i, ch in enumerate(text):
        if i in ignored:
            continue
        low = ch.translate(SAME).lower()
        # ⚠️ One character in, one out: a few characters lowercase to two, which would shift every index after
        ch = low if len(low) == 1 else ch
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        where.append(i)
    return "".join(chars), where


def _sentence_end(text: str, end: int) -> int:
    """Where a matched sentence really ends: past its closing emphasis, its citations and its full stop, and a
    citation written after the stop ("... events. [2]"), all of which go with it."""
    def past_marks(i):
        while True:
            group = GROUP.match(text, i)
            if group:
                i = group.end()
            elif i < len(text) and text[i] == "*":
                i += 1
            else:
                return i
    end = past_marks(end)
    if end < len(text) and text[end] in ".!?":
        end = past_marks(end + 1)
    return end


def apply_edits(answer: str, edits: List[Dict]) -> Tuple[str, Dict]:
    """Apply each edit in order and return the edited answer and a report of what was applied and what failed."""
    report = {"edits": len(edits), "applied": [], "failed": []}
    for item in edits:
        sentence, text, action = str(item.get("sentence", "")), str(item.get("text", "")).strip(), item.get("action")
        wanted = _normalized(sentence)[0].strip().rstrip(".!?;: ")
        if action not in ACTIONS or not wanted or not text:
            report["failed"].append({**item, "why": "malformed"})
            continue
        haystack, where = _normalized(answer)
        count = haystack.count(wanted)
        if count != 1:
            # ⚠️ A sentence found twice is left alone: changing the wrong one would be worse than changing neither
            report["failed"].append({**item, "why": "not found" if count == 0 else "ambiguous"})
            continue
        at = haystack.index(wanted)
        start, end = where[at], _sentence_end(answer, where[at + len(wanted) - 1] + 1)
        if action == "replace":
            # Bold that opened before the sentence goes too, or the new sentence would start with a stray "**"
            while start > 0 and answer[start - 1] == "*":
                start -= 1
            answer = answer[:start] + text + answer[end:]
        else:
            answer = answer[:end] + " " + text + answer[end:]
        report["applied"].append(item)
    return answer, report
