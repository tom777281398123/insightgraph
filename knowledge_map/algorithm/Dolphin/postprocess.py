import argparse
import json
import os
import re
import shutil
from collections import Counter
from PIL import Image
import numpy as np


def is_blank_image(image_path, white_threshold=250, white_ratio_threshold=0.95):

    try:
        img = Image.open(image_path).convert('L')  # 转灰度
        img_array = np.array(img)
        white_pixels = np.sum(img_array > white_threshold)
        total_pixels = img_array.size
        white_ratio = white_pixels / total_pixels
        return white_ratio > white_ratio_threshold
    except Exception as e:
        print(f"检查图片失败 {image_path}: {e}")
        return False


def detect_repetitive_text(text, min_pattern_length=5, repetition_threshold=5):
    
    if not text or len(text) < min_pattern_length * repetition_threshold:
        return False
    
    # 清理文本
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # 寻找重复模式
    for pattern_len in range(min_pattern_length, min(50, len(clean_text) // 3)):
        for start in range(len(clean_text) - pattern_len * 2):
            pattern = clean_text[start:start + pattern_len]
            if not pattern.strip():
                continue
            
            # 计算模式重复次数
            count = clean_text.count(pattern)
            if count >= repetition_threshold:
                # 验证这个重复确实占据了文本的大部分
                if len(pattern) * count > len(clean_text) * 0.5:
                    return True
    
    return False


def is_hallucination_element(element):
    
    text = element.get('text', '')
    label = element.get('label', '')
    
    # 空文本
    if not text or len(text.strip()) == 0:
        return True
    
    # 检测重复模式
    if detect_repetitive_text(text):
        return True
    
    # 检测常见的幻觉模式
    hallucination_patterns = [
        r'^[\s↑.\[\]0-9\-]+$',  # 只有箭头、点、日期等无意义符号
        r'(\d{4}-\d{2}-\d{2}\s*\[\d{4}-\d{2}-\d{2}\]\.?\s*){3,}',  # 重复日期模式
        r'(GBIF\s*\.\s*\[\d{4}-\d{2}-\d{2}\]\.?\s*){3,}',  # GBIF重复模式
        r'(↑\s*\.\s*){3,}',  # 重复的箭头点模式
        r'(\$\$[^$]+\$\$\s*){3,}',  # 重复的数学公式
    ]
    
    for pattern in hallucination_patterns:
        if re.search(pattern, text):
            return True
    
    # 检测无意义的短文本（只有数字或符号）
    clean_text = re.sub(r'[\s\$\{\}\\_\[\]\(\)\|\-\.\,\:\;\!\?\'\"\^\#\@\&\*\+\=\/\\]', '', text)
    if len(clean_text) <= 2 and len(text) > 5:
        return True
    
    # 检测数学公式幻觉（多个不相关的公式片段）
    if text.count('$$') >= 4:
        # 如果有很多独立的公式块，可能是幻觉
        formula_blocks = re.findall(r'\$\$[^$]+\$\$', text)
        if len(formula_blocks) >= 3:
            # 检查公式内容是否有意义
            meaningful_formulas = 0
            for formula in formula_blocks:
                # 去掉公式标记后的内容长度
                content = re.sub(r'[\$\s\\]', '', formula)
                if len(content) > 5:
                    meaningful_formulas += 1
            if meaningful_formulas < len(formula_blocks) * 0.5:
                return True
    
    return False


def calculate_meaningful_content_ratio(elements):
    
    if not elements:
        return 0.0
    
    total_elements = len(elements)
    meaningful_elements = 0
    
    for elem in elements:
        if not is_hallucination_element(elem):
            meaningful_elements += 1
    
    return meaningful_elements / total_elements


def is_blank_page(elements, save_dir=None, meaningful_threshold=0.3):
    
    if not elements:
        return True, "无元素"
    
    # 计算有意义内容比例
    meaningful_ratio = calculate_meaningful_content_ratio(elements)
    if meaningful_ratio < meaningful_threshold:
        return True, f"有意义内容比例过低 ({meaningful_ratio:.1%})"
    
    # 检查所有figure是否都是空白的
    figure_elements = [e for e in elements if e.get('label') == 'fig']
    blank_figures = 0
    if figure_elements and save_dir:
        for fig in figure_elements:
            fig_path = fig.get('figure_path', '')
            if fig_path:
                full_path = os.path.join(save_dir, 'markdown', fig_path)
                if os.path.exists(full_path) and is_blank_image(full_path):
                    blank_figures += 1
        
        # 如果所有figure都是空白的，且没有其他有意义内容
        if blank_figures == len(figure_elements):
            non_fig_elements = [e for e in elements if e.get('label') != 'fig']
            if not non_fig_elements or all(is_hallucination_element(e) for e in non_fig_elements):
                return True, "所有图片为空白且无其他有效内容"
    
    # 额外检查：如果大部分元素都是幻觉，也标记为空白页
    non_fig_elements = [e for e in elements if e.get('label') != 'fig']
    if non_fig_elements:
        hallucination_count = sum(1 for e in non_fig_elements if is_hallucination_element(e))
        if hallucination_count / len(non_fig_elements) > 0.7:
            # 还需要检查figure是否也是空白的
            if not figure_elements or (figure_elements and blank_figures > 0):
                return True, f"大部分文本元素为幻觉 ({hallucination_count}/{len(non_fig_elements)})"
    
    return False, ""


def postprocess_results(input_json_path, output_dir=None, meaningful_threshold=0.3, 
                        remove_blank_pages=True, remove_hallucinations=True):
    
    # 读取输入
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确定保存目录
    input_dir = os.path.dirname(os.path.dirname(input_json_path))  # 上级目录
    save_dir = input_dir
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'recognition_json'), exist_ok=True)
    
    # 统计信息
    stats = {
        'total_pages': 0,
        'blank_pages': 0,
        'blank_page_numbers': [],
        'hallucination_elements_removed': 0,
        'pages_processed': 0
    }
    
    processed_pages = []
    
    # 处理每一页
    pages = data.get('pages', [])
    stats['total_pages'] = len(pages)
    
    for page in pages:
        page_number = page.get('page_number', 0)
        elements = page.get('elements', [])
        
        # 检测空白页
        is_blank, reason = is_blank_page(elements, save_dir, meaningful_threshold)
        
        if is_blank and remove_blank_pages:
            stats['blank_pages'] += 1
            stats['blank_page_numbers'].append(page_number)
            print(f"  页面 {page_number}: 检测为空白/无效页面 - {reason}")
            # 添加空页面标记
            processed_pages.append({
                'page_number': page_number,
                'elements': [],
                'is_blank': True,
                'blank_reason': reason
            })
            continue
        
        # 过滤幻觉元素
        if remove_hallucinations:
            filtered_elements = []
            for elem in elements:
                if is_hallucination_element(elem):
                    stats['hallucination_elements_removed'] += 1
                else:
                    filtered_elements.append(elem)
            elements = filtered_elements
        
        processed_pages.append({
            'page_number': page_number,
            'elements': elements
        })
        stats['pages_processed'] += 1
    
    # 构建处理后的数据
    processed_data = {
        'source_file': data.get('source_file', ''),
        'total_pages': data.get('total_pages', len(pages)),
        'processing_stats': stats,
        'pages': processed_pages
    }
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    output_json_path = os.path.join(output_dir, 'recognition_json', f'{base_name}_processed.json')
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 生成处理后的Markdown
    generate_processed_markdown(processed_data, output_dir, base_name)
    
    return processed_data, stats, output_json_path


def generate_processed_markdown(data, output_dir, base_name):
    
    os.makedirs(os.path.join(output_dir, 'markdown'), exist_ok=True)
    
    markdown_content = []
    
    for page in data.get('pages', []):
        page_number = page.get('page_number', 0)
        elements = page.get('elements', [])
        is_blank = page.get('is_blank', False)
        
        if is_blank:
            markdown_content.append(f"\n\n<!-- 第 {page_number} 页: 空白页 -->\n\n---\n")
            continue
        
        if page_number > 1:
            markdown_content.append("\n\n---\n\n")
        
        for elem in elements:
            text = elem.get('text', '')
            label = elem.get('label', '')
            
            if label == 'fig':
                markdown_content.append(f"\n{text}\n")
            elif label == 'tab':
                markdown_content.append(f"\n{text}\n")
            elif label in ['sec_1', 'sec_0']:
                markdown_content.append(f"\n## {text}\n")
            elif label == 'sec_2':
                markdown_content.append(f"\n### {text}\n")
            elif label == 'cap':
                markdown_content.append(f"\n*{text}*\n")
            else:
                markdown_content.append(f"\n{text}\n")
    
    # 保存Markdown
    markdown_path = os.path.join(output_dir, 'markdown', f'{base_name}_processed.md')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(''.join(markdown_content))
    
    print(f"  处理后的Markdown已保存: {markdown_path}")


def main():
    parser = argparse.ArgumentParser(description='OCR结果后处理工具')
    parser.add_argument('--input', '-i', required=True, 
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录（默认为输入目录下的processed子目录）')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='有意义内容比例阈值 (默认0.3)')
    parser.add_argument('--keep-blank', action='store_true',
                        help='保留空白页（不移除）')
    parser.add_argument('--keep-hallucinations', action='store_true',
                        help='保留幻觉元素（不过滤）')
    
    args = parser.parse_args()
    
    print(f"\n开始后处理: {args.input}")
    print(f"有意义内容阈值: {args.threshold}")
    
    processed_data, stats, output_path = postprocess_results(
        args.input,
        args.output,
        meaningful_threshold=args.threshold,
        remove_blank_pages=not args.keep_blank,
        remove_hallucinations=not args.keep_hallucinations
    )
    
    print(f"\n处理完成!")
    print(f"  总页数: {stats['total_pages']}")
    print(f"  空白/无效页数: {stats['blank_pages']}")
    if stats['blank_page_numbers']:
        print(f"  空白页页码: {stats['blank_page_numbers']}")
    print(f"  移除的幻觉元素数: {stats['hallucination_elements_removed']}")
    print(f"  有效页数: {stats['pages_processed']}")
    print(f"  结果保存至: {output_path}")


if __name__ == '__main__':
    main()
