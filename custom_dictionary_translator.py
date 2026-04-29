from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

def load_dictionary(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    eng, chi = parts
                    dictionary[eng.strip().lower()] = chi.strip()
                else:
                    print(f"警告: 忽略格式不正确的行: {line}")
            elif line:
                print(f"警告: 忽略没有冒号的行: {line}")
    return dictionary

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

def split_text(text):
    pattern = re.compile(r'([\u4e00-\u9fff]+)')
    parts = pattern.split(text)
    return [part for part in parts if part]

def preprocess(text, dictionary):
    words = text.split()
    i = 0
    while i < len(words):
        if any(is_chinese(char) for char in words[i]):
            i += 1
            continue
        if words[i].lower() in dictionary:
            words[i] = dictionary[words[i].lower()]
        else:
            for j in range(3, 0, -1):
                if i + j <= len(words):
                    phrase = ' '.join(words[i:i+j]).lower()
                    if phrase in dictionary:
                        words[i:i+j] = [dictionary[phrase]]
                        break
        i += 1
    return ' '.join(words)

def postprocess(translated_text, dictionary):
    reverse_dict = {v: k for k, v in dictionary.items()}
    for chi, eng in reverse_dict.items():
        translated_text = translated_text.replace(chi, eng)
    return translated_text

def translate_english_to_chinese(english_text, tokenizer, model, dictionary):
    parts = split_text(english_text)
    translated_parts = []

    for part in parts:
        if any(is_chinese(char) for char in part):
            translated_parts.append(part)
        else:
            preprocessed_text = preprocess(part, dictionary)
            print("预处理后的文本:", preprocessed_text)

            inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("机器翻译结果:", translated_text)

            postprocessed_text = postprocess(translated_text, dictionary)
            translated_parts.append(postprocessed_text)

    return ''.join(translated_parts)

if __name__ == '__main__':
    # 加载模型和分词器
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 加载自定义字典
    dictionary_path = "./custom_dictionary.txt"
    translation_dict = load_dictionary(dictionary_path)

    # 测试翻译
    english_text = "Hello, how are you? I am learning machine learning."
    chinese_text = translate_english_to_chinese(english_text, tokenizer, model, translation_dict)
    print("最终翻译结果:", chinese_text)

    # 添加更多测试用例
    test_cases = [
        "Machine learning is a subset of artificial intelligence.",
        "I love machine learning and deep learning.",
        "The quick brown fox jumps over the lazy dog."
    ]

    for case in test_cases:
        result = translate_english_to_chinese(case, tokenizer, model, translation_dict)
        print(f"\n原文: {case}")
        print(f"翻译: {result}")