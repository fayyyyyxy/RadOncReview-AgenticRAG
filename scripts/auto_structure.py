import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned" / "cleaned_text_breast.json"

OUTPUT_FILE = BASE_DIR / "data" / "structured" / "structured_breast_sample.json"


def extract_dose(text: str):
    """Extract dose in Gy, e.g. '60 Gy'."""
    m = re.search(r"(\d+)\s*gy", text, flags=re.I)
    return m.group(1) if m else None


def extract_fractions(text: str):
    """Extract number of fractions, e.g. '30 fractions' or '30 fx'."""
    m = re.search(r"(\d+)\s*(fractions|fx)", text, flags=re.I)
    return m.group(1) if m else None


def extract_chemo(text: str):
    """Detect common chemotherapy agents in the text."""
    drugs = ["cisplatin", "carboplatin", "paclitaxel", "etoposide"]
    found = []
    for d in drugs:
        if re.search(rf"\b{d}\b", text, flags=re.I):
            found.append(d.capitalize())
    return sorted(set(found))


def get_heading_from_node(node: dict) -> str:
    """
    A node may have heading_1 / heading_2 / heading_3.
    We return whichever exists (in order of specificity).
    """
    return (
        node.get("heading_3")
        or node.get("heading_2")
        or node.get("heading_1")
        or ""
    )


def structure_node(node: dict, heading_path: list[str]):
    """
    Turn a single node into a structured record.

    heading_path = list of headings from root to this node, e.g.:
      ["Breast", "DCIS"]

    We join them into a single 'heading_path' string for context.
    """
    # Current level heading (if any)
    current_heading = get_heading_from_node(node)
    path = heading_path + ([current_heading] if current_heading else [])

    full_text = (node.get("content") or "").strip()
    dose = extract_dose(full_text)
    fractions = extract_fractions(full_text)
    chemo = extract_chemo(full_text)

    return {
        "heading": current_heading,
        "heading_path": " > ".join([h for h in path if h]),
        "full_text": full_text,
        "dose_gy": dose,
        "fractions": fractions,
        "chemo": chemo,
    }


def walk_nodes(node_list, heading_path=None):
    """
    Recursively walk through a list of nodes, collecting structured records.
    Each node may also contain its own 'children'.
    """
    if heading_path is None:
        heading_path = []

    records = []

    for node in node_list:
        if not isinstance(node, dict):
            continue

        # Create record for this node (even if content is short)
        rec = structure_node(node, heading_path)
        records.append(rec)

        # Recurse into children if present
        child_nodes = node.get("children", [])
        if isinstance(child_nodes, list) and child_nodes:
            # For children, extend the heading path with current heading
            current_heading = get_heading_from_node(node)
            new_path = heading_path + ([current_heading] if current_heading else [])
            records.extend(walk_nodes(child_nodes, new_path))

    return records


def main():
    print("Reading:", INPUT_FILE)

    if not INPUT_FILE.exists():
        print("ERROR: Input file does not exist.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level structure:
    # { "Category": "...", "children": [ {...}, {...}, ... ] }
    top_children = data.get("children", [])
    print(f"Top-level children: {len(top_children)}")

    structured_records = walk_nodes(top_children, heading_path=[])

    print(f"Total structured records (including heading_2, etc.): {len(structured_records)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_records, f, indent=2, ensure_ascii=False)

    print(f"Saved structured output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()