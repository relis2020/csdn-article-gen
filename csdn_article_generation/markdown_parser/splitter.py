#splitter.py
import re

def build_full_heading_path(headings, max_length=100):
    """根据 headings 列表构造完整路径名，限制总长度"""
    if not headings:
        return "文档"
    
    # 确保 headings 中的每个元素都有 'heading' 键
    heading_texts = []
    for h in headings:
        if isinstance(h, dict) and 'heading' in h:
            heading_texts.append(h['heading'])
        elif isinstance(h, str):
            heading_texts.append(h)
    
    if not heading_texts:
        return "文档"
        
    path = " > ".join(heading_texts)
    
    if len(path) > max_length:
        # 保留前面的部分，截断后面
        return path[:max_length-3] + "..."
    
    return path

def get_content_prefix(content, max_length=30):
    """
    智能提取内容前缀用于命名
    - 跳过无意义的开头句子
    - 限制总长度
    """
    if not content:
        return ""
    
    # 移除标题行
    lines = content.strip().splitlines()
    if not lines:
        return ""
    
    # 过滤掉标题行
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # 跳过标题行 (# 开头的行)
        if not line_stripped.startswith('#'):
            filtered_lines.append(line_stripped)
    
    if not filtered_lines:
        return ""
    
    # 常见的无意义开头句子模式
    meaningless_patterns = [
        r'^\s*本[章节节]\s*(将)?\s*(介绍|讲述|说明|阐述|描述)',
        r'^\s*[介讲说阐描]述',
        r'^\s*在本[章节节]中',
        r'^\s*我们将',
        r'^\s*首先',
        r'^\s*接下来',
        r'^\s*[因此所以然后]',
        r'^\s*[-—–]\s*',  # 破折号开头
        r'^\s*\*\s*',     # 星号开头
        r'^\s*NOTE\s*[-—–]?', # NOTE 开头
        r'^\s*NOTE—',     # NOTE— 开头
    ]
    
    # 尝试从第一行开始找有意义的内容
    for line in filtered_lines:
        if not line:
            continue
            
        # 跳过无意义的句子
        is_meaningless = False
        for pattern in meaningless_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_meaningless = True
                break
        
        if is_meaningless:
            continue
            
        # 清理特殊字符，只保留中文、英文、数字
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', line).strip()
        if len(cleaned) < 3:  # 太短的也不合适
            continue
            
        # 限制长度
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned
    
    # 如果都没找到合适的，就用第一行
    first_line = filtered_lines[0] if filtered_lines else ""
    if first_line:
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', first_line).strip()
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned
    
    return ""

def get_content_suffix(content, max_length=30):
    """
    智能提取内容后缀用于命名
    - 跳过无意义的结尾句子
    - 限制总长度
    """
    if not content:
        return ""
    
    # 移除标题行
    lines = content.strip().splitlines()
    if not lines:
        return ""
    
    # 过滤掉标题行
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # 跳过标题行 (# 开头的行)
        if not line_stripped.startswith('#'):
            filtered_lines.append(line_stripped)
    
    if not filtered_lines:
        return ""
    
    # 常见的无意义结尾句子模式
    meaningless_patterns = [
        r'^\s*总结.*',
        r'^\s*小结.*',
        r'^\s*综上所述.*',
        r'^\s*总而言之.*',
        r'^\s*以上.*',
        r'^\s*最后.*',
        r'^\s*结束语.*',
        r'^\s*参考文献.*',
        r'^\s*致谢.*',
        r'^\s*附录.*',
        r'^\s*[-—–]\s*$',  # 单独的破折号
        r'^\s*\*\s*$',     # 单独的星号
    ]
    
    # 尝试从最后一行开始找有意义的内容
    for line in reversed(filtered_lines):
        if not line:
            continue
            
        # 跳过无意义的句子
        is_meaningless = False
        for pattern in meaningless_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_meaningless = True
                break
        
        if is_meaningless:
            continue
            
        # 清理特殊字符，只保留中文、英文、数字
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', line).strip()
        if len(cleaned) < 3:  # 太短的也不合适
            continue
            
        # 限制长度
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned
    
    # 如果都没找到合适的，就用最后一行
    last_line = filtered_lines[-1] if filtered_lines else ""
    if last_line:
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', last_line).strip()
        if len(cleaned) > max_length:
            return cleaned[:max_length] + "..."
        return cleaned
    
    return ""

def build_paraname(headings, content="", max_total_length=150):
    """
    构造完整的 paraname，包含路径和内容前缀+后缀
    """
    full_path = build_full_heading_path(headings)
    
    if not content:
        return full_path
    
    content_prefix = get_content_prefix(content)
    content_suffix = get_content_suffix(content)
    
    if content_prefix and content_suffix and content_prefix != content_suffix:
        paraname = f"{full_path} - {content_prefix}...{content_suffix}"
    elif content_prefix:
        paraname = f"{full_path} - {content_prefix}"
    elif content_suffix:
        paraname = f"{full_path} - ...{content_suffix}"
    else:
        paraname = full_path
    
    # 限制总长度
    if len(paraname) > max_total_length:
        return paraname[:max_total_length-3] + "..."
    
    return paraname

def build_section_headings_tree(outline, section_position, section_heading, section_level):
    """
    根据目录大纲构建当前章节的完整路径（从根到当前章节）
    """
    if not outline or section_heading is None:
        return [{'heading': section_heading}] if section_heading else []
    
    # 找到当前章节在大纲中的位置
    current_section_index = -1
    for i, item in enumerate(outline):
        if item['position'] == section_position and item['level'] == section_level and item['title'] == section_heading:
            current_section_index = i
            break
    
    if current_section_index == -1:
        return [{'heading': section_heading}]
    
    # 构建从根到当前章节的完整路径
    path = []
    
    # 收集从根到当前章节的所有层级
    for i in range(current_section_index + 1):
        item = outline[i]
        # 只添加当前路径上的章节（层级递增的）
        if not path or item['level'] > path[-1]['level']:
            path.append({
                'heading': item['title'],
                'level': item['level'],
                'position': item['position']
            })
        # 如果找到同级或更高级的章节，替换路径
        elif item['level'] <= section_level:
            # 找到应该插入的位置
            while path and path[-1]['level'] >= item['level']:
                path.pop()
            path.append({
                'heading': item['title'],
                'level': item['level'],
                'position': item['position']
            })
    
    # 确保路径是连续的（从1级开始递增）
    if path:
        cleaned_path = []
        expected_level = 1
        for item in path:
            if item['level'] == expected_level:
                cleaned_path.append(item)
                expected_level += 1
            elif item['level'] < expected_level:
                # 重新调整
                while cleaned_path and cleaned_path[-1]['level'] >= item['level']:
                    cleaned_path.pop()
                cleaned_path.append(item)
                expected_level = item['level'] + 1
        
        # 只保留到当前章节的路径
        final_path = []
        for item in cleaned_path:
            final_path.append(item)
            if item['level'] == section_level and item['heading'] == section_heading:
                break
        
        return final_path
    
    return [{'heading': section_heading}]

def merge_headings_path(existing_headings, new_heading):
    """
    合并两个章节路径，保持正确的层级顺序
    """
    if not existing_headings:
        return [new_heading] if new_heading else []
    
    if not new_heading:
        return existing_headings
    
    # 简单合并，确保不重复
    result = existing_headings.copy()
    new_heading_info = {
        'heading': new_heading['heading'] if isinstance(new_heading, dict) else new_heading,
        'level': new_heading['level'] if isinstance(new_heading, dict) and 'level' in new_heading else 0,
        'position': new_heading['position'] if isinstance(new_heading, dict) and 'position' in new_heading else 0
    }
    
    # 检查是否已存在
    exists = False
    for h in result:
        if h['heading'] == new_heading_info['heading']:
            exists = True
            break
    
    if not exists:
        result.append(new_heading_info)
    
    return result

def split_long_section(section, max_split_length):
    """分割超长段落"""
    content = section['content']
    paragraphs = re.split(r'\n\s*\n', content)  # 按空行分割段落
    result = []
    current_chunk = ''

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # 如果当前段落本身超过最大长度，需要进一步拆分
        if len(paragraph) > max_split_length:
            # 如果当前块不为空，先加入结果
            if current_chunk:
                result.append(current_chunk)
                current_chunk = ''
                
            # 按句子拆分
            sentences = re.findall(r'[^.!?。！？\n]+[.!?。！？]*', paragraph)
            if not sentences:
                sentences = [paragraph]
                
            sentence_chunk = ''
            for sentence in sentences:
                if len(sentence_chunk + sentence) <= max_split_length:
                    sentence_chunk += sentence
                else:
                    if sentence_chunk:
                        result.append(sentence_chunk)
                    # 如果单个句子超过最大长度，按固定长度分割
                    if len(sentence) > max_split_length:
                        for i in range(0, len(sentence), max_split_length):
                            result.append(sentence[i:i + max_split_length])
                        sentence_chunk = ''
                    else:
                        sentence_chunk = sentence
            
            if sentence_chunk:
                result.append(sentence_chunk)
                
        elif len(current_chunk + '\n\n' + paragraph) <= max_split_length:
            # 如果添加当前段落不超过最大长度，则添加到当前块
            current_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
        else:
            # 如果添加当前段落超过最大长度，则将当前块加入结果，并重新开始一个新块
            if current_chunk:
                result.append(current_chunk)
            current_chunk = paragraph

    # 添加最后一个块（如果有）
    if current_chunk:
        result.append(current_chunk)
        
    return result

def process_sections(sections, outline, min_split_length=1500, max_split_length=2000):
    """
    处理段落，根据最小和最大分割字数进行分割
    """
    # 为每个 section 构建完整的 headings 路径
    for section in sections:
        if section['heading'] and 'headings' not in section:
            section['headings'] = build_section_headings_tree(outline, section['position'], section['heading'], section['level'])

    # 预处理：将相邻的小段落合并
    preprocessed_sections = []
    current_section = None

    for section in sections:
        content_length = len(section['content'].strip())

        if content_length < min_split_length and current_section:
            # 如果当前段落小于最小长度且有累积段落，尝试合并
            heading_text = ''
            if section['heading']:
                heading_text = f"{'#' * section['level']} {section['heading']}\n"
            
            merged_content = f"{current_section['content']}\n\n{heading_text}{section['content']}"

            if len(merged_content) <= max_split_length:
                # 如果合并后不超过最大长度，则合并
                current_section['content'] = merged_content
                if section['heading']:
                    # 合并 headings 路径
                    if 'headings' not in current_section:
                        current_section['headings'] = []
                    # 添加当前章节到路径中
                    current_section['headings'] = build_section_headings_tree(outline, section['position'], section['heading'], section['level'])
                continue

        # 如果无法合并，则开始新的段落
        if current_section:
            preprocessed_sections.append(current_section)
        current_section = section

    # 添加最后一个段落
    if current_section:
        preprocessed_sections.append(current_section)

    result = []
    accumulated_section = None  # 用于累积小于最小分割字数的段落

    for i in range(len(preprocessed_sections)):
        section = preprocessed_sections[i]
        content_length = len(section['content'].strip())

        # 检查是否需要累积段落
        if content_length < min_split_length:
            # 如果还没有累积过段落，创建新的累积段落
            if not accumulated_section:
                accumulated_section = {
                    'heading': section['heading'],
                    'level': section['level'],
                    'content': section['content'],
                    'position': section['position'],
                    'headings': section.get('headings', [])
                }
            else:
                # 已经有累积段落，将当前段落添加到累积段落中
                heading_text = ''
                if section['heading']:
                    heading_text = f"{'#' * section['level']} {section['heading']}\n"
                
                accumulated_section['content'] += f"\n\n{heading_text}{section['content']}"
                if section['heading']:
                    # 更新 headings 路径
                    accumulated_section['headings'] = build_section_headings_tree(outline, section['position'], section['heading'], section['level'])

            # 只有当累积内容达到最小长度时才处理
            accumulated_length = len(accumulated_section['content'].strip())
            if accumulated_length >= min_split_length:
                if accumulated_length > max_split_length:
                    # 如果累积段落超过最大长度，进一步分割
                    sub_sections = split_long_section(accumulated_section, max_split_length)
                    
                    for j, sub_content in enumerate(sub_sections):
                        # 使用改进的 paraname 构造函数
                        paraname = build_paraname(accumulated_section.get('headings', []), sub_content)
                            
                        result.append({
                            "filename": "input.md",
                            "paraname": paraname,
                            "content": sub_content
                        })
                else:
                    # 添加到结果中
                    paraname = build_paraname(accumulated_section.get('headings', []), accumulated_section['content'])
                    result.append({
                        "filename": "input.md",
                        "paraname": paraname,
                        "content": accumulated_section['content']
                    })

                accumulated_section = None  # 重置累积段落
            continue

        # 如果有累积的段落，先处理它
        if accumulated_section:
            accumulated_length = len(accumulated_section['content'].strip())
            
            if accumulated_length > max_split_length:
                # 如果累积段落超过最大长度，进一步分割
                sub_sections = split_long_section(accumulated_section, max_split_length)
                
                for j, sub_content in enumerate(sub_sections):
                    paraname = build_paraname(accumulated_section.get('headings', []), sub_content)
                    result.append({
                        "filename": "input.md",
                        "paraname": paraname,
                        "content": sub_content
                    })
            else:
                # 添加到结果中
                paraname = build_paraname(accumulated_section.get('headings', []), accumulated_section['content'])
                result.append({
                    "filename": "input.md",
                    "paraname": paraname,
                    "content": accumulated_section['content']
                })

            accumulated_section = None  # 重置累积段落

        # 处理当前段落
        if content_length > max_split_length:
            # 如果段落长度超过最大分割字数，需要进一步分割
            sub_sections = split_long_section(section, max_split_length)

            for j, sub_content in enumerate(sub_sections):
                # 使用改进的 paraname 构造函数
                paraname = build_paraname(section.get('headings', []), sub_content)
                result.append({
                    "filename": "input.md",
                    "paraname": paraname,
                    "content": sub_content
                })
        else:
            # 直接添加到结果
            heading_text = ''
            if section['heading']:
                heading_text = f"{'#' * section['level']} {section['heading']}\n"
            
            content = f"{heading_text}{section['content']}"
            
            # 使用改进的 paraname 构造函数
            paraname = build_paraname(section.get('headings', []), section['content'])
            result.append({
                "filename": "input.md",
                "paraname": paraname,
                "content": content
            })

    # 处理最后剩余的小段落
    if accumulated_section:
        accumulated_length = len(accumulated_section['content'].strip())
        if result and accumulated_length <= (max_split_length - len(result[-1]['content'])):
            # 尝试与最后一个结果合并
            heading_text = ''
            if accumulated_section['heading']:
                heading_text = f"{'#' * accumulated_section['level']} {accumulated_section['heading']}\n"
            
            merged_content = f"{result[-1]['content']}\n\n{heading_text}{accumulated_section['content']}"
            
            if len(merged_content) <= max_split_length:
                paraname = build_paraname(result[-1].get('headings', []), merged_content)
                result[-1] = {
                    "filename": "input.md",
                    "paraname": paraname,
                    "content": merged_content
                }
            else:
                # 作为单独段落添加
                heading_content = f"{'#' * accumulated_section['level']} {accumulated_section['heading']}\n{accumulated_section['content']}" if accumulated_section['heading'] else accumulated_section['content']
                paraname = build_paraname(accumulated_section.get('headings', []), accumulated_section['content'])
                result.append({
                    "filename": "input.md",
                    "paraname": paraname,
                    "content": heading_content
                })
        else:
            # 直接添加
            heading_content = f"{'#' * accumulated_section['level']} {accumulated_section['heading']}\n{accumulated_section['content']}" if accumulated_section['heading'] else accumulated_section['content']
            paraname = build_paraname(accumulated_section.get('headings', []), accumulated_section['content'])
            result.append({
                "filename": "input.md",
                "paraname": paraname,
                "content": heading_content
            })

    return result
