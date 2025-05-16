# Text Summarization Tool

This is a Python-based text summarization tool that uses the BART (Bidirectional and Auto-Regressive Transformers) model to generate concise summaries of long texts.

## Features

- Uses state-of-the-art BART model for summarization
- Handles long texts by automatically splitting them into chunks
- Supports both file input/output and direct text input
- Customizable summary length
- Easy to use command-line interface

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can use the tool in several ways:

1. Summarize text from a file:
```bash
python text_summarizer.py --input input.txt --output summary.txt
```

2. Summarize text from standard input:
```bash
python text_summarizer.py
```
Then type or paste your text and press Ctrl+D when finished.

3. Customize summary length:
```bash
python text_summarizer.py --input input.txt --max_length 200 --min_length 100
```

## Parameters

- `--input`: Path to the input text file (optional)
- `--output`: Path to save the summary (optional)
- `--max_length`: Maximum length of the summary (default: 150)
- `--min_length`: Minimum length of the summary (default: 50)

## Example

```bash
python text_summarizer.py --input article.txt --output summary.txt
```

This will read the text from `article.txt`, generate a summary, and save it to `summary.txt`.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NLTK
- NumPy

## Note

The first time you run the script, it will download the BART model and NLTK data, which might take a few minutes depending on your internet connection. 