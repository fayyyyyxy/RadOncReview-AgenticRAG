def extract_hierarchical_sections(doc_json, disease):
    """ Formats the Google Doc JSON to follow the hirerarchy of headings and subheadings. """
    all_sections = []

    def add_section(sections_stack, new_section):
        """Attach new section at the correct hierarchy level."""
        # If new section is deeper (subheading)
        if sections_stack and new_section["level"] > sections_stack[-1]["level"]:
            sections_stack[-1]["children"].append(new_section)
        else:
            # Pop until we find a parent or root
            while sections_stack and new_section["level"] <= sections_stack[-1]["level"]:
                sections_stack.pop()
            if sections_stack:
                sections_stack[-1]["children"].append(new_section)
            else:
                all_sections.append(new_section)
        sections_stack.append(new_section)

    # Traverse tabs
    for tab in doc_json.get("tabs", []):
        document_tab = tab.get("documentTab")
        if not document_tab:
            continue

        sections_stack = []
        body = document_tab.get("body", {})

        for content in body.get("content", []):
            # --- Paragraphs ---
            if "paragraph" in content:
                paragraph = content["paragraph"]
                style = paragraph.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
                text = "".join(
                    elem.get("textRun", {}).get("content", "")
                    for elem in paragraph.get("elements", [])
                    if "textRun" in elem
                ).strip()

                if not text:
                    continue

                if style.startswith("HEADING_"):
                    level = int(style.split("_")[-1])
                    new_section = {
                        "heading": text,
                        "level": level,
                        "content_parts": [],
                        "children": []
                    }
                    add_section(sections_stack, new_section)
                else:
                    # Add text to current section if exists
                    if sections_stack:
                        sections_stack[-1]["content_parts"].append(text)
                    else:
                        # content before any heading
                        all_sections.append({
                            "heading": None,
                            "level": 0,
                            "content_parts": [text],
                            "children": []
                        })

            # --- Tables ---
            elif "table" in content:
                table = content["table"]
                for row in table.get("tableRows", []):
                    for cell in row.get("tableCells", []):
                        for cell_content in cell.get("content", []):
                            if "paragraph" in cell_content:
                                cell_text = "".join(
                                    elem.get("textRun", {}).get("content", "")
                                    for elem in cell_content["paragraph"].get("elements", [])
                                    if "textRun" in elem
                                ).strip()
                                if cell_text and sections_stack:
                                    sections_stack[-1]["content_parts"].append(cell_text)

    def clean_section(sec):
        """Convert content_parts to single text and recursively clean children."""
        clean_content = " ".join(sec.get("content_parts", [])).replace("\n", " ").strip()
        cleaned = {
            f"heading_{sec['level']}": sec["heading"],
            "content": clean_content
        }
        if sec["children"]:
            cleaned["children"] = [clean_section(c) for c in sec["children"]]
        return cleaned
        
    # get rid of meaningless heading_0 parts
    cleaned_sections = [clean_section(s) for s in all_sections if s["level"] > 0]
    
    document = {
        "Category": f"{disease}",
        "children": cleaned_sections
    }
    return document