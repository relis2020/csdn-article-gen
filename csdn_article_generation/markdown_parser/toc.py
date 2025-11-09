#toc.py
import re

def extract_outline(text):
    pattern = r'^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$'
    matches = re.finditer(pattern, text, re.MULTILINE)
    outline = []
    for match in matches:
        level = len(match.group(1))
        title = match.group(2).strip()
        position = match.start()
        outline.append({
            'level': level,
            'title': title,
            'position': position
        })
    return outline

def toc_to_text(outline):
    lines = []
    for item in outline:
        indent = '  ' * (item['level'] - 1)
        lines.append(f"{indent}- {item['title']}")
    return '\n'.join(lines)
