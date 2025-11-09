#main.py
import sys
import json
import os
from toc import extract_outline, toc_to_text
from parser import split_by_headings
from splitter import process_sections

def main(file_path, min_length=500, max_length=800):
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 提取目录大纲
    outline = extract_outline(text)
    
    # 按标题分割文档
    sections = split_by_headings(text, outline)
    
    # 处理段落分割
    blocks = process_sections(sections, outline, min_length, max_length)

    # ===== 修改1：保存目录树到文件 =====
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    toc_file_path = f"{filename_without_ext}_toc.txt"
    
    with open(toc_file_path, 'w', encoding='utf-8') as f:
        f.write("目录树：\n")
        f.write(toc_to_text(outline))
    
    print(f"目录树已保存到: {toc_file_path}")

    # ===== 修改2：保存JSON块到文件 =====
    json_file_path = f"{filename_without_ext}_blocks.json"
    
    # 设置正确的filename
    filename = os.path.basename(file_path)
    for block in blocks:
        block['filename'] = filename
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)
    
    print(f"JSON块已保存到: {json_file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <file.md> [min_length] [max_length]")
        print("默认值: min_length=500, max_length=800")
        sys.exit(1)
    
    file_path = sys.argv[1]
    min_length = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    
    main(file_path, min_length, max_length)
