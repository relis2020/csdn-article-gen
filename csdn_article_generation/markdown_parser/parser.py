#parser.py
def split_by_headings(text, outline):
    if not outline:
        return [{'heading': None, 'level': 0, 'content': text, 'position': 0}]

    sections = []

    # 添加第一个标题前的内容
    if outline[0]['position'] > 0:
        front_matter = text[:outline[0]['position']].strip()
        if front_matter:
            sections.append({
                'heading': None,
                'level': 0,
                'content': front_matter,
                'position': 0
            })

    for i in range(len(outline)):
        current = outline[i]
        next_pos = outline[i+1]['position'] if i+1 < len(outline) else len(text)

        heading_line_end = text.find('\n', current['position'])
        if heading_line_end == -1:
            heading_line_end = len(text)
        content_start = heading_line_end + 1

        content = text[content_start:next_pos].strip()

        sections.append({
            'heading': current['title'],
            'level': current['level'],
            'content': content,
            'position': current['position']
        })

    return sections
