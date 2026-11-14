# Sources: docs/docs demo/model.md; docs/docs demo/lab_3_description.md
"""Exercise 2, bước 3: tải model và tokenizer đã huấn luyện sẵn. Đặc tả: docs/docs demo/model.md."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exercise_2.common import config_from_args, update_report

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config["model_name"])


def load_model(config):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )


def count_parameters(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main():
    config = config_from_args(__doc__)
    model = load_model(config)
    parameters = count_parameters(model)
    update_report(config, "model", {"name": config["model_name"], "parameters": parameters})
    print(model.pre_classifier, model.dropout, model.classifier, sep="\n")
    print(f"Tổng tham số: {parameters['total']:,} | Có thể huấn luyện: {parameters['trainable']:,}")


if __name__ == "__main__":
    main()
