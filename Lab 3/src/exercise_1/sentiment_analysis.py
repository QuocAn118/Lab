# Sources: docs/lab_3_description.md; docs/model.md; docs/demo.md
"""Exercise 1: tokenize một câu và phân tích cảm xúc bằng model có sẵn trên Hugging Face Hub."""
import argparse

from transformers import AutoTokenizer, pipeline

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"


def validate_text(text):
    if not text.strip():
        raise ValueError("Câu đầu vào không được rỗng.")
    return text


def analyze(text):
    validate_text(text)
    tokens = AutoTokenizer.from_pretrained(MODEL_NAME).tokenize(text)
    result = pipeline("sentiment-analysis", model=MODEL_NAME)(text)[0]
    return {"sentence": text, "tokens": tokens, "label": result["label"], "score": result["score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Câu cần phân tích")
    result = analyze(parser.parse_args().text)
    print("Sentence:", result["sentence"])
    print("Tokens:", result["tokens"])
    print("Sentiment:", result["label"])
    print(f"Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()
