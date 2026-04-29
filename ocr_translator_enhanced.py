import re

import cv2
import pytesseract
from transformers import pipeline,AutoModelWithLMHead,AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
import textwrap
# pytesseract.pytesseract.tesseract_cmd = r'path\to\tesseract.exe'  # Uncomment and set your Tesseract path if needed


# 提取文本
def get_text_from_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return None

    # 显示图像供用户选择 ROI
    cv2.imshow('Select ROI', image)
    roi = cv2.selectROI('Select ROI', image)
    cv2.destroyWindow('Select ROI')

    if roi == (0, 0, 0, 0):
        print("No region selected.")
        return None

    # 提取选定的 ROI
    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w]

    # 使用 Tesseract OCR 识别文本
    text = pytesseract.image_to_string(cropped_image)
    return text.strip()

# 翻译
def translate_text(text, source_language='zh',dest_language='en'):
    print(f'正在加载【{source_language} - {dest_language}】翻译模型.')

    if source_language == 'zh' and dest_language == 'en':
        model_name = 'Helsinki-NLP/opus-mt-zh-en'
    elif source_language == 'en' and dest_language == 'zh':
        model_name = 'Helsinki-NLP/opus-mt-en-zh'
    elif source_language == 'de' and dest_language == 'zh':
        model_name = 'Helsinki-NLP/opus-mt-de-zh'
    elif source_language == 'zh' and dest_language == 'de':
        model_name = 'Helsinki-NLP/opus-mt-zh-de'
    else:
        print("Model for this language pair not found.")
        return None
    print('模型加载完成'.center(60,'-'))
    print('开始翻译'.center(60,'-'))
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation = pipeline(f'translation_{source_language}_to_{dest_language}', model=model, tokenizer=tokenizer)
    print('翻译完成'.center(60,'-'))
    return translation(text)[0]['translation_text']

def modelLoad():
    print('正在加载【汉语-英语】翻译模型.')
    model = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-en')
    translation = pipeline('translation_zh_to_en', model=model, tokenizer=tokenizer)
    print('正在加载【英语-汉语】翻译模型.')
    model_en2zh = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    tokenizer_en2zh = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-zh')
    translation_en2zh = pipeline('translation_en_to_zh', model=model_en2zh, tokenizer = tokenizer_en2zh)
    print('正在加载【德语-汉语】翻译模型.')
    model_de2zh = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-de-ZH')
    tokenizer_de2zh = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-ZH')
    translation_de2zh = pipeline('translation_de_to_zh', model=model_de2zh, tokenizer=tokenizer_de2zh)
    print('正在加载【汉语 - 德语】翻译模型.')
    model_zh2de = AutoModelWithLMHead.from_pretrained('Helsinki-NLP/opus-mt-zh-de')
    tokenizer_zh2de = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-zh-de')
    translation_zh2de = pipeline('translation_zh_to_de', model=model_zh2de, tokenizer=tokenizer_zh2de)

    print('模型加载完成'.center(60,'-'))

def customPrint(text, max_width):
    """
    将给定的文本按照最大宽度分隔并逐行打印。
    :param text: 要打印的文本
    :param max_width: 每行的最大字符数
    """
    wrapper = textwrap.TextWrapper(width=max_width)
    wrapped_lines = wrapper.wrap(text)

    for line in wrapped_lines:
        print(line)

def clean_text(text):
    # 定义常见拼写错误及其正确的形式
    spelling_corrections = {
        "fs": "as",  # 明显错误
        "@": " ",     # 移除不适当的符号
        "‘":' '
    }
    # 替换拼写错误
    for error, correction in spelling_corrections.items():
        text = text.replace(error, correction)
    # 移除非字母数字字符，保留空格和基本标点符号
    text = re.sub(r'[^A-Za-z0-9\s\.,;!?-]', '', text)
    # 确保多个空格被合并为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    image_path = 'path/to/your/image.png'  # 替换为你的图片路径
    text = get_text_from_image(image_path)
    if text:
        n = 60 # 每行显示的最大字符数 n
        customPrint((f"检测到的文本为: {clean_text(text)}"),n)
        translated_text = translate_text(text, source_language='en',dest_language='zh')  # 目标语言更改为中文
        if translated_text:
            customPrint(f"翻译后的文本为: {translated_text}", n)