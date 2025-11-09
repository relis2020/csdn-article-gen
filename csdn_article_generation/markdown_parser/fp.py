import json
import requests
import re
import sys
import signal

# 全局变量用于处理中断
interrupted = False

def signal_handler(signum, frame):
    """处理Ctrl+C中断信号"""
    global interrupted
    print("\n\n收到中断信号，正在优雅退出...")
    interrupted = True

def extract_image_urls(content):
    """从内容中提取图片URL"""
    # 匹配Markdown图片语法 ![alt](url)
    markdown_pattern = r'!\[.*?\]\((.*?)\)'
    # 匹配HTML img标签
    html_pattern = r'<img[^>]*src=["\'](.*?)["\'][^>]*>'
    
    markdown_urls = re.findall(markdown_pattern, content)
    html_urls = re.findall(html_pattern, content)
    
    return markdown_urls + html_urls

def process_image_with_api(image_url, api_key):
    """使用API分析图片并转换为Mermaid格式"""
    global interrupted
    if interrupted:
        return None
        
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    payload = {
        "model": "glm-4.5v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url.strip()}
                    },
                    {
                        "type": "text",
                        "text": "Please analyze this technical image and convert it to standard mermaid chart code. Requirements: 1. If it's a flowchart, convert to mermaid flowchart format 2. If it's an architecture diagram, convert to mermaid graph format 3. If it's a sequence diagram, convert to mermaid sequenceDiagram format 4. If it's a Gantt chart, convert to mermaid gantt format 5. Preserve all technical details and relationships exactly as shown 6. Use the original labels and text from the image 7. Return only mermaid code, no other text 8. Wrap code with ```mermaid"
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
        return None
    except Exception as e:
        if not interrupted:
            print(f"API调用错误: {e}")
        return None

def extract_mermaid_code(api_response):
    """从API响应中提取Mermaid代码"""
    if not api_response:
        return None
    
    # 提取mermaid代码块
    pattern = r'```mermaid\s*(.*?)\s*```'
    match = re.search(pattern, api_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def convert_image_to_mermaid_format(image_url, mermaid_code):
    """将图片转换为Mermaid格式"""
    if mermaid_code:
        # 创建Mermaid图表格式
        mermaid_format = f"""
<div align="center">

```mermaid
{mermaid_code}
```

<p><em>Original image: {image_url}</em></p>
</div>
"""
        return mermaid_format
    else:
        # 如果转换失败，返回原图片
        return f"![Image]({image_url})\n<p><em>Chart conversion failed, original image retained</em></p>"

def replace_images_with_mermaid(content, image_url, mermaid_description):
    """将图片替换为Mermaid图表"""
    # 替换Markdown图片语法
    markdown_pattern = r'!\[.*?\]\(' + re.escape(image_url) + r'\)'
    
    # 替换HTML img标签
    html_pattern = r'<img[^>]*src=["\']' + re.escape(image_url) + r'["\'][^>]*>'
    
    # 优先替换Markdown格式
    if re.search(markdown_pattern, content):
        content = re.sub(markdown_pattern, mermaid_description, content)
    elif re.search(html_pattern, content):
        content = re.sub(html_pattern, mermaid_description, content)
    else:
        # 如果没有找到原格式，在图片URL后添加描述
        content = content.replace(image_url, f"{image_url}\n\n{mermaid_description}")
    
    return content

def process_json_file(input_file, output_file, api_key):
    """处理JSON文件，将图片转换为Mermaid图表"""
    global interrupted
    print(f"开始处理文件: {input_file}")
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return False
    
    total_items = len(data)
    print(f"共 {total_items} 个项目需要处理")
    print("按 Ctrl+C 可以中断处理...")
    
    processed_count = 0
    # 处理每个条目
    for i, item in enumerate(data):
        if interrupted:
            print(f"\n处理被中断，已完成 {processed_count}/{total_items} 个项目")
            break
            
        content = item.get('content', '')
        if content:
            # 提取图片URL
            image_urls = extract_image_urls(content)
            
            if image_urls:
                print(f"处理第 {i+1}/{total_items} 项，包含 {len(image_urls)} 张图片")
                
                # 处理每张图片
                for j, image_url in enumerate(image_urls):
                    if interrupted:
                        break
                        
                    print(f"  处理图片 {j+1}/{len(image_urls)}: {image_url}")
                    
                    # 使用API获取Mermaid代码
                    api_response = process_image_with_api(image_url, api_key)
                    if interrupted:
                        break
                        
                    mermaid_code = extract_mermaid_code(api_response)
                    
                    if mermaid_code:
                        # 转换为Mermaid格式
                        mermaid_format = convert_image_to_mermaid_format(image_url, mermaid_code)
                        
                        # 替换图片为Mermaid图表
                        content = replace_images_with_mermaid(content, image_url, mermaid_format)
                        print(f"  ✓ 图片转换为Mermaid图表完成")
                    else:
                        # 转换失败，保留原图
                        fallback = convert_image_to_mermaid_format(image_url, None)
                        content = replace_images_with_mermaid(content, image_url, fallback)
                        print(f"  ✗ Mermaid转换失败，保留原图")
            
            # 更新内容
            item['content'] = content
            processed_count = i + 1
    
    # 保存处理后的JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"处理完成，结果保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存输出文件失败: {e}")
        return False

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("使用方法: python fp.py input.json output.json")
        print("参数说明:")
        print("  input.json   - 输入的JSON文件路径")
        print("  output.json  - 输出的JSON文件路径")
        sys.exit(1)
    
    # 配置参数
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    api_key = "e4a74940778a480d93a41fffbba58237.TkeUIruGAxmgYtPD"
    
    # 处理文件
    success = process_json_file(input_file, output_file, api_key)
    if success:
        print("程序正常结束")
    else:
        print("程序执行出错")
        sys.exit(1)

if __name__ == "__main__":
    main()
