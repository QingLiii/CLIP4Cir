import json
import os
import re
import requests
import concurrent.futures
from tqdm import tqdm

# DeepSeek API配置
API_KEY = "sk-bb523cca272a45e5ba04e38a968c7dc7"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 数据路径配置
DATA_ROOT = "E:\\Github\\CLIP4Cir\\books_set\\books_set"
CAPTIONS_FILE = os.path.join(DATA_ROOT, "captions.json")
OUTPUT_FILE = os.path.join(DATA_ROOT, "captions_processed.json")

# 最大token限制
MAX_TOKENS = 77

def clean_caption(caption):
    """
    清理caption中的冗余内容
    1. 去除开头的空格
    2. 去除形如"A, ..."或"B, ..."的前缀
    3. 去除括号中的引用，如"(B)"或"(Fig. 1)"
    """
    # 去除开头的空格
    caption = caption.strip()
    
    # 去除形如"A, ..."或"B, ..."的前缀
    caption = re.sub(r'^[A-Z],\s+', '', caption)
    
    # 去除括号中的引用，如"(B)"或"(Fig. 1)"
    caption = re.sub(r'\([A-Z]\)', '', caption)
    
    # 去除形如"B, the..."的前缀
    caption = re.sub(r'^[A-Z],\s+the\s+', 'The ', caption)
    
    # 确保首字母大写
    if caption and caption[0].islower():
        caption = caption[0].upper() + caption[1:]
    
    return caption

def compress_with_deepseek(text):
    """
    使用DeepSeek API压缩文本至指定token数量以内
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""请将以下医学图像描述压缩至77个tokens以内，如果内容小于77个tokens请不要做任何修改，只压缩和大于77tokens的内容，保留核心医学内容和关键术语。请直接返回压缩后的文本，不要添加任何额外解释。

原文：{text}

压缩后："""
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        compressed_text = result["choices"][0]["message"]["content"].strip()
        return compressed_text
    except Exception as e:
        print(f"API调用失败: {e}")
        # 如果API调用失败，返回原文本
        return text

def process_single_caption(key_item):
    """
    处理单个caption的函数，用于并发处理
    """
    key, item = key_item
    # 复制原始数据
    processed_item = item.copy()
    
    # 清理caption
    cleaned_caption = clean_caption(item['caption'])
    
    # 压缩caption
    compressed_caption = compress_with_deepseek(cleaned_caption)
    
    # 更新处理后的caption
    processed_item['caption'] = compressed_caption
    processed_item['original_caption'] = item['caption']  # 保存原始caption
    
    return key, processed_item

def process_captions():
    """
    处理captions.json文件，清理并压缩caption，使用并发处理
    """
    # 读取原始数据
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建新的数据字典
    processed_data = {}
    
    # 设置并发数量
    max_workers = 20
    
    # 处理每个caption，使用并发
    print(f"正在处理{len(data)}个caption，使用{max_workers}个并发线程...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_key = {executor.submit(process_single_caption, (key, item)): key for key, item in data.items()}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(data)):
            try:
                key, processed_item = future.result()
                processed_data[key] = processed_item
            except Exception as exc:
                print(f"处理caption时出错: {exc}")
    
    # 保存处理后的数据
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存至 {OUTPUT_FILE}")

def main():
    """
    主函数
    """
    print("开始处理books数据集...")
    process_captions()
    print("处理完成！")

if __name__ == "__main__":
    main()