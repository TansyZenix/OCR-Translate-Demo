# OCR Translate Demo

An image text extraction (OCR) and translation tool using **Tesseract OCR** and **Helsinki-NLP OPUS-MT** translation models.

## Features

- **OCR Text Extraction** — Select a region from an image and extract text using Tesseract OCR
- **Machine Translation** — Translate extracted text between Chinese, English, and German using Hugging Face OPUS-MT models
- **Custom Dictionary** — Enhance translation quality with a user-defined glossary for domain-specific terms

## Project Structure

| File | Description |
|------|-------------|
| `ocr_translator.py` | Basic OCR → translation pipeline (supports zh/en/de) |
| `ocr_translator_enhanced.py` | Enhanced version with OCR text cleaning / spell correction |
| `custom_dictionary_translator.py` | Translation with pre/post-processing using a custom dictionary |
| `custom_dictionary.txt` | Sample custom dictionary (English → Chinese glossary) |

## Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- Python packages (install via `pip`):
  ```
  pip install opencv-python pytesseract transformers torch torchvision
  ```

## Quick Start

### 1. Install Tesseract OCR

- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

If Tesseract is not in your system PATH, uncomment and set the path in the script:
```python
# pytesseract.pytesseract.tesseract_cmd = r'path\to\tesseract.exe'
```

### 2. Run OCR + Translation

```bash
python ocr_translator.py
```

- A window will open showing your image
- Drag to select the region of text
- The extracted text will be translated and printed

### 3. Run with Custom Dictionary

```bash
python custom_dictionary_translator.py
```

Edit `custom_dictionary.txt` to add your own term mappings (format: `english_term:中文翻译`).

## Supported Language Pairs

- Chinese ↔ English
- Chinese ↔ German

## License

MIT
