import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import argparse

def download_nltk_data():
    """Download required NLTK data"""
    nltk.download('punkt')

def summarize_text(text, max_length=150, min_length=50):
    """
    Summarize the input text using a pre-trained model
    
    Args:
        text (str): Input text to summarize
        max_length (int): Maximum length of the summary
        min_length (int): Minimum length of the summary
    
    Returns:
        str: Generated summary
    """
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Split text into chunks if it's too long
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        current_length += len(sentence.split())
        if current_length > 1024:  # BART's maximum input length
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence.split())
        else:
            current_chunk.append(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)

def main():
    parser = argparse.ArgumentParser(description='Text Summarization Tool')
    parser.add_argument('--input', type=str, help='Input text file path')
    parser.add_argument('--output', type=str, help='Output summary file path')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum length of summary')
    parser.add_argument('--min_length', type=int, default=50, help='Minimum length of summary')
    
    args = parser.parse_args()
    
    # Download required NLTK data
    download_nltk_data()
    
    if args.input:
        # Read from file
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Read from stdin
        print("Enter your text (press Ctrl+D when finished):")
        text = ""
        try:
            while True:
                line = input()
                text += line + "\n"
        except EOFError:
            pass
    
    # Generate summary
    summary = summarize_text(text, args.max_length, args.min_length)
    
    if args.output:
        # Write to file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
    else:
        # Print to stdout
        print("\nGenerated Summary:")
        print(summary)

if __name__ == "__main__":
    main() 