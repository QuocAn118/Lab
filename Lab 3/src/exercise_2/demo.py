# Sources: docs/demo.md; docs/lab_3_description.md
"""Suy luận nhanh bằng mốc `best` với câu thực tế. Đặc tả: docs/demo.md."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exercise_2.common import config_from_args, project_path, update_report

# Câu trung tính không có nhãn đúng vì model chỉ có 2 lớp, nên chỉ quan sát, không tính đúng/sai.
SENTENCES = [
    ("This movie was an absolute masterpiece! The acting and plot were brilliant.", "POSITIVE"),
    ("Waste of time. The acting was terrible and the storyline made no sense.", "NEGATIVE"),
    ("It was an average film, neither great nor awful.", None),
]


def main():
    config = config_from_args(__doc__)
    best_dir = project_path(config["paths"]["checkpoints_dir"]) / "best"
    if not best_dir.exists():
        raise SystemExit(f"Không thấy {best_dir}. Hãy chạy exercise_2.training trước.")
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).eval()

    results = []
    for sentence, truth in SENTENCES:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=config["data"]["max_length"]
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
        predicted = probabilities.argmax().item()
        label = model.config.id2label[predicted]
        confidence = probabilities[predicted].item()
        results.append({
            "sentence": sentence,
            "prediction": label,
            "confidence": confidence,
            "truth": truth,
            "correct": None if truth is None else label == truth,
        })
        print(f"Câu: {sentence}\n-> {label} ({confidence:.2%}), nhãn thật: {truth}\n")
    update_report(config, "demo", results)


if __name__ == "__main__":
    main()
