"""
Citation Check: after the answer is written, ask the LLM whether each cited passage states what its sentence says.
A citation it rejects is removed, so a claim the passages lack is never credited to them. The sentence stays and
is listed in the report for an interface to mark: a note inside the answer text lowered judged completeness.
A reference list after a "References:" line is not checked; its entries follow the citations left in the body.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from .llm_client import BaseLLMClient


# One bracket: [3], [4, 5] or [1-3]; the writer is told to use separate brackets but often does not
BRACKET = r"\[\d+(?:\s*[,–-]\s*\d+)*\]"
# Neighbouring brackets, "[4], [5]", "[4] and [5]" or "[4][5]", form one group, so a removed citation takes the comma
# or "and" before it along. The leading spaces are captured so that an emptied group takes them along too
GROUP = re.compile(rf"([ \t]*)({BRACKET}(?:(?:[ \t]*,)?(?:[ \t]*and)?[ \t]*{BRACKET})*)")
# A "References:" line, also **References** or ## Sources; the writer copies the passages' "[n] Source: ..." below it
REFERENCES = re.compile(r"^[ \t]*(?:#+[ \t]*)?\**[ \t]*(?:references|sources)[ \t]*\**[ \t]*:?[ \t]*\**[ \t]*$",
                        re.IGNORECASE | re.MULTILINE)
# 💡 A decimal point is followed by a digit, not a space, so "2.5 Gy" never ends a sentence
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def numbers(group: str) -> List[int]:
    """Citation numbers in a group, in order: "[4, 5]" -> [4, 5]; "[1-3]" -> [1, 2, 3]; "[4], [6]" -> [4, 6]."""
    found = []
    # ⚠️ Split at closing brackets as well as commas: "[4][7]" read as one part would be the range 4-7
    for part in re.split(r"[,\]]", group):
        ends = [int(n) for n in re.findall(r"\d+", part)]
        span = range(ends[0], ends[-1] + 1) if len(ends) == 2 and ends[0] < ends[1] else ends
        found.extend(n for n in span if n not in found)
    return found


def cited(answer: str) -> Set[int]:
    """Every citation number in an answer, whichever bracket style it uses."""
    return {n for m in GROUP.finditer(answer) for n in numbers(m.group(2))}


def split_references(answer: str) -> Tuple[str, str]:
    """The body of an answer, and the reference list from its last "References:" line on ("" when there is none)."""
    headings = list(REFERENCES.finditer(answer))
    return (answer[:headings[-1].start()], answer[headings[-1].start():]) if headings else (answer, "")


def _sentence_start(answer: str, position: int) -> int:
    start = 0
    for end in SENTENCE_END.finditer(answer, 0, position):
        start = end.end()
    return start


def sentence_of(answer: str, start: int, end: int) -> str:
    """The sentence a citation at answer[start:end] belongs to."""
    left = _sentence_start(answer, start)
    # A citation written after the full stop ("... events. [2]") belongs to the sentence it follows
    while left > 0 and not GROUP.sub("", answer[left:start]).strip():
        left = _sentence_start(answer, left - 1)
    stop = SENTENCE_END.search(answer, end)
    return answer[left:stop.end() if stop else len(answer)].strip()


class CitationChecker:
    """Checks every (sentence, cited passage) pair of an answer in one LLM call."""

    SYSTEM_PROMPT = ("You check whether cited passages support the sentences that cite them. "
                     "Judge only what the passage says, not whether the sentence is true.")
    MAX_ATTEMPTS = 2  # one retry when the reply cannot be used

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    def check(self, answer: str, passages: List[Dict]) -> Tuple[str, Dict]:
        """Return the answer without rejected citations, and a report of what was done."""
        text = {p["citation_id"]: p["text"] for p in passages}
        # The reference list is a key to the numbers, not a claim: "[1] Source: ..." is never supported by passage 1
        body, references = split_references(answer)
        groups = list(GROUP.finditer(body))
        items = {}  # (sentence, passage number) -> item number in the prompt
        for group in groups:
            sentence = sentence_of(body, group.start(), group.end())
            for n in numbers(group.group(2)):
                # A number with no passage behind it is rejected without asking
                if n in text and (sentence, n) not in items:
                    items[(sentence, n)] = len(items) + 1
        report = {"citations": sum(len(numbers(g.group(2))) for g in groups), "items": len(items),
                  "unchecked": 0, "removed": 0, "attempts": 0, "error": None, "uncited_sentences": []}
        verdicts, report["error"], report["attempts"] = self._ask(items, text) if items else ({}, None, 0)
        if report["error"]:
            # ⚠️ Fail open: a failed check must not delete citations, so the answer is returned unchanged
            report["unchecked"] = len(items)
            return answer, report
        report["unchecked"] = sum(1 for i in items.values() if i not in verdicts)

        parts, cut_at, last, length = [], [], 0, 0
        for group in groups:
            sentence = sentence_of(body, group.start(), group.end())
            listed = numbers(group.group(2))
            # Kept unless rejected: an item the LLM did not answer stays cited (fail open again)
            kept = [n for n in listed if n in text and verdicts.get(items[(sentence, n)]) is not False]
            report["removed"] += len(listed) - len(kept)
            # Kept numbers are written one per bracket: "[4], [5]" or "[4, 5]" becomes "[4] [5]".
            # ⚠️ An emptied group drops its leading space too, or "tolerated [2]." would become "tolerated ."
            replacement = (group.group(1) + " ".join(f"[{n}]" for n in kept)) if kept else ""
            parts += [body[last:group.start()], replacement]
            length += group.start() - last
            if not kept:
                cut_at.append(length)  # where the bracket was, in the checked answer
            length += len(replacement)
            last = group.end()
        checked = "".join(parts) + body[last:]
        # 💡 dict.fromkeys drops repeats but keeps order, unlike set()
        report["uncited_sentences"] = list(dict.fromkeys(sentence_of(checked, p, p) for p in cut_at))
        # An entry stays while the body still cites its number, so a removed citation does not return through it
        still_cited = cited(checked)
        checked += "".join(line for line in references.splitlines(keepends=True)
                           if not cited(line) or cited(line) & still_cited)
        return checked, report

    def _ask(self, items: Dict[Tuple[str, int], int],
             text: Dict[int, str]) -> Tuple[Dict[int, bool], Optional[str], int]:
        """Item number -> supported, an error message when no reply could be used, and the number of attempts."""
        passages = "\n\n".join(f"[{n}]\n{text[n]}" for n in sorted({n for _, n in items}))
        lines = "\n".join(f"{i}. Passage [{n}] | Sentence: {sentence}" for (sentence, n), i in items.items())
        prompt = f"""For each numbered item, decide whether the passage states what the sentence says where it \
cites that passage.
Answer "yes" only if the passage supports that statement, including every number in it (dose, fractions, \
percentages, years).
Answer "no" if the passage is about something else, says something different, or only shares the topic.

## Passages
{passages}

## Items
{lines}"""
        error = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            # 💡 The retry names the problem, so it is a new request rather than the same one again at temperature 0
            request = prompt if error is None else (
                f"{prompt}\n\nYour previous reply could not be used ({error}). Reply with the JSON object only.")
            try:
                reply = self.llm.generate_structured(
                    request,
                    output_schema={"verdicts": [{"item": "<item number>", "supported": "<yes or no>"}]},
                    system_prompt=self.SYSTEM_PROMPT
                )
            except Exception as e:
                # Not retried here: the AWS client already retries throttling and connection errors
                return {}, f"{type(e).__name__}: {e}", attempt
            if "error" not in reply and isinstance(reply.get("verdicts"), list):
                break
            error = str(reply.get("error", "reply has no verdicts list"))
        else:
            return {}, error, self.MAX_ATTEMPTS
        verdicts = {}
        for verdict in reply["verdicts"]:
            if not isinstance(verdict, dict):
                continue
            value = str(verdict.get("supported", "")).strip().lower()
            # ⚠️ Only a clear yes or no counts; anything else leaves the item unanswered
            if str(verdict.get("item", "")).strip().isdigit() and value in ("yes", "no", "true", "false"):
                verdicts[int(str(verdict["item"]).strip())] = value in ("yes", "true")
        return verdicts, None, attempt
