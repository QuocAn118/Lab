"""Exercise 2, bước 2 và 4: tải IMDB, chia tập và tiền xử lý. Đặc tả: docs/data.md."""
from collections import Counter

from datasets import load_dataset

from exercise_2.common import config_from_args, update_report


def clean_text(text):
    # IMDB chứa thẻ <br /> sẽ bị tách thành token vô nghĩa và chiếm chỗ trong max_length.
    return text.replace("<br />", " ")


def split_dataset(train_source, test_source, train_size, val_size, test_size, seed):
    if train_size + val_size > len(train_source) or test_size > len(test_source):
        raise ValueError("Số mẫu yêu cầu lớn hơn số mẫu của dataset.")
    shuffled = train_source.shuffle(seed=seed)
    train = shuffled.select(range(train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = test_source.shuffle(seed=seed).select(range(test_size))
    return train, val, test


def load_splits(config):
    data = config["data"]
    source = load_dataset(data["dataset_name"])
    return split_dataset(
        source["train"], source["test"], data["train_size"], data["val_size"],
        data["test_size"], config["seed"],
    )


def tokenize_dataset(dataset, tokenizer, config):
    def tokenize(batch):
        texts = [clean_text(text) for text in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=config["data"]["max_length"])

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def main():
    config = config_from_args(__doc__)
    train, val, test = load_splits(config)
    summary = {}
    for name, split in {"train": train, "val": val, "test": test}.items():
        labels = Counter(split["label"])
        summary[name] = {"samples": len(split), "negative": labels[0], "positive": labels[1]}
    # Chia theo chỉ số nên không trùng mẫu, nhưng IMDB gốc có thể có review trùng nội dung.
    train_texts, val_texts, test_texts = set(train["text"]), set(val["text"]), set(test["text"])
    summary["duplicate_texts"] = {
        "train_val": len(train_texts & val_texts),
        "train_val_vs_test": len((train_texts | val_texts) & test_texts),
    }
    update_report(config, "data", summary)
    for name, values in summary.items():
        print(name, values)
    print("\nMẫu đầu tiên của tập train:", clean_text(train[0]["text"])[:300], "...")


if __name__ == "__main__":
    main()
