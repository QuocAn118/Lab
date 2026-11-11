import torch 
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
from lab_2.common import (
    argument_parser,
    load_config,
    project_path,
    update_report,
)
from PIL import Image

CLASS_NAMES = ["airplane",    # Nhãn 0
               "automobile",  # Nhãn 1
               "bird",        # Nhãn 2
               "cat",        # Nhãn 3
               "deer",        # Nhãn 4
               "dog",         # Nhãn 5
               "frog",        # Nhãn 6
               "horse",       # Nhãn 7
               "ship",        # Nhãn 8
               "truck",       # Nhãn 9
               ]

def split_indices(total, train_size, val_size, seed):
    if train_size + val_size > total: 
        raise ValueError("The number of requested images exceeds the number of available images")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator= generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    return train_indices, val_indices

def build_transform(config, train):
    data_config = config["data"]

    steps = [] # Create a list of processing steps

    if train:
        steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.RandomCrop(32, padding= data_config["crop_padding"]))
    steps.extend([transforms.Resize((data_config["image_size"], 
                                     data_config["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean= data_config["mean"],
                                     std= data_config["std"]),])

    return transforms.Compose(steps)

def load_splits(config):
    data_config = config["data"]
    root = project_path(data_config["data_dir"])

    train_dataset = datasets.CIFAR10(root= root,
                                     train= True, 
                                     download= True,
                                     transform= build_transform(config, train= True),)

    val_dataset = datasets.CIFAR10(root= root, 
                                   train= True,
                                   download= True,
                                   transform= build_transform(config, train= False),)

    test_dataset = datasets.CIFAR10(root= root,
                                    train= False, 
                                    download= True,
                                    transform= build_transform(config, train= False),)

    train_indices, val_indices = split_indices(total= 50000,
                                                  train_size= data_config["train_size"],
                                                  val_size= data_config["val_size"],
                                                  seed= config["seed"],)
    
    test_indices, _ = split_indices(total= 10000,
                                      train_size= data_config["test_size"],
                                      val_size= 0,
                                      seed= config["seed"],)

    train = Subset(train_dataset, train_indices)
    test = Subset(test_dataset, test_indices)
    val = Subset(val_dataset, val_indices)

    return train, val, test

def make_loader(dataset, batch_size, config, shuffle=False):
    generator = None

    if shuffle: 
        generator = torch.Generator().manual_seed(config["seed"])

    return DataLoader(dataset, 
                      batch_size= batch_size,
                      shuffle= shuffle, 
                      num_workers= config["data"]["num_workers"],
                      pin_memory= torch.cuda.is_available(),
                      generator= generator,)

def raw_samples(subset, count):
    samples = []

    for index in subset.indices[:count]:
        image = Image.fromarray(subset.dataset.data[index])
        label = subset.dataset.targets[index]
        samples.append((image, label))

    return samples

def describe(subset):
    class_counts = {name: 0 for name in CLASS_NAMES}

    for index in subset.indices:
        label = subset.dataset.targets[index]
        class_name = CLASS_NAMES[label]
        class_counts[class_name] += 1

    return {"samples": len(subset),
            "class_counts": class_counts,}

def main():
    args = argument_parser(__doc__).parse_args()
    config = load_config(args.config)

    train, val, test = load_splits(config)

    assert set(train.indices).isdisjoint(val.indices), (
        "Tập train và validation bị trùng chỉ số ảnh."
    )

    report = {"train": describe(train),
              "val": describe(val),
              "test": describe(test),}

    update_report(config, "data", report)

    print(f"Train: {len(train)} ảnh")
    print(f"Validation: {len(val)} ảnh")
    print(f"Test: {len(test)} ảnh")

    image, label = train[0]
    print(f"Shape ảnh sau tiền xử lý: {tuple(image.shape)}")

if __name__ == "__main__":
    main()