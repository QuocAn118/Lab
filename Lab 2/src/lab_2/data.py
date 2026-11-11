"""Bước 5: CIFAR-10, chia tập cố định và transforms theo cấu hình main."""
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lab_2.common import argument_parser, load_config, project_path, update_report

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def split_indices(total, train_size, val_size, seed):
    if min(total, train_size, val_size) < 0 or train_size + val_size > total:
        raise ValueError("Số mẫu không hợp lệ hoặc lớn hơn dataset")
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[:train_size], indices[train_size:train_size + val_size]


def build_transform(config, train):
    data = config["data"]
    steps = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=data["crop_padding"])] if train else []
    return transforms.Compose(steps + [
        transforms.Resize((data["image_size"], data["image_size"])),
        transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"]),
    ])


def load_splits(config):
    data = config["data"]
    root = project_path(data["data_dir"])
    train = datasets.CIFAR10(root, train=True, download=True, transform=build_transform(config, True))
    val = datasets.CIFAR10(root, train=True, download=True, transform=build_transform(config, False))
    test = datasets.CIFAR10(root, train=False, download=True, transform=build_transform(config, False))
    train_ids, val_ids = split_indices(len(train), data["train_size"], data["val_size"], config["seed"])
    test_ids, _ = split_indices(len(test), data["test_size"], 0, config["seed"])
    return Subset(train, train_ids), Subset(val, val_ids), Subset(test, test_ids)


def make_loader(dataset, batch_size, config, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config["data"]["num_workers"], pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config["seed"]) if shuffle else None,
    )


def raw_samples(subset, count):
    if count < 0:
        raise ValueError("Số ảnh phải không âm")
    return [(Image.fromarray(subset.dataset.data[i]), int(subset.dataset.targets[i]))
            for i in subset.indices[:count]]


def describe(subset):
    counts = Counter(int(subset.dataset.targets[i]) for i in subset.indices)
    return {"samples": len(subset), "class_counts": {name: counts[i] for i, name in enumerate(CLASS_NAMES)}}


def main():
    config = load_config(argument_parser(__doc__).parse_args().config)
    train, val, test = load_splits(config)
    assert not set(train.indices) & set(val.indices), "Train và val bị trùng mẫu"
    summary = {name: describe(subset) for name, subset in [("train", train), ("val", val), ("test", test)]}
    update_report(config, "data", summary)
    print(summary)
    print("Input shape:", tuple(train[0][0].shape))


if __name__ == "__main__":
    main()
