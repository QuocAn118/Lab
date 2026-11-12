"""CIFAR-10 theo docs/data.md §5; phụ thuộc ghép TV3, Codex 2026-11-11."""
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lab_2.common import argument_parser, load_config, project_path, update_report

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def split_indices(total, train_size, val_size, seed):
    if min(total, train_size, val_size) < 0 or train_size + val_size > total:
        raise ValueError("Invalid split sizes")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:train_size], order[train_size:train_size + val_size]


def build_transform(config, train):
    cfg = config["data"]
    steps = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=cfg["crop_padding"])] if train else []
    return transforms.Compose(steps + [transforms.Resize((cfg["image_size"], cfg["image_size"])),
                                      transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])


def load_splits(config):
    cfg = config["data"]
    root = str(project_path(cfg["data_dir"]))
    sources = [datasets.CIFAR10(root, train=training, download=True,
                               transform=build_transform(config, augment))
               for training, augment in ((True, True), (True, False), (False, False))]
    train, val = split_indices(len(sources[0]), cfg["train_size"], cfg["val_size"], config["seed"])
    test, _ = split_indices(len(sources[2]), cfg["test_size"], 0, config["seed"])
    return tuple(Subset(source, indices) for source, indices in zip(sources, (train, val, test)))


def make_loader(dataset, batch_size, config, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      generator=torch.Generator().manual_seed(config["seed"]) if shuffle else None,
                      num_workers=config["data"]["num_workers"], pin_memory=torch.cuda.is_available())


def raw_samples(subset, count):
    return [(Image.fromarray(subset.dataset.data[i]), int(subset.dataset.targets[i]))
            for i in subset.indices[:count]]


def describe(subset):
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for index in subset.indices:
        counts[CLASS_NAMES[int(subset.dataset.targets[index])]] += 1
    return {"samples": len(subset), "class_counts": counts}


def main():
    config = load_config(argument_parser(__doc__).parse_args().config)
    train, val, test = load_splits(config)
    assert not set(train.indices) & set(val.indices)
    report = {name: describe(subset) for name, subset in zip(("train", "val", "test"), (train, val, test))}
    update_report(config, "data", report)
    print(report)
    print("Image shape:", train[0][0].shape)


if __name__ == "__main__":
    main()
