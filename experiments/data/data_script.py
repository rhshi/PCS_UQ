from datasets import load_dataset
import os

if __name__ == "__main__":
    os.makedirs(os.path.dirname("./data/cifar100/"), exist_ok=True)
    os.makedirs(os.path.dirname("./data/tinyimagenet/"), exist_ok=True)
    os.makedirs(os.path.dirname("./data/caltechbirds/"), exist_ok=True)

    load_dataset("uoft-cs/cifar100", split="train").save_to_disk("./data/cifar100/train")
    load_dataset("uoft-cs/cifar100", split="test").save_to_disk("./data/cifar100/test")

    load_dataset("zh-plus/tiny-imagenet", split="train").save_to_disk("./data/tinyimagenet/train")
    load_dataset("zh-plus/tiny-imagenet", split="valid").save_to_disk("./data/tinyimagenet/test")

    load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="train").save_to_disk("./data/caltechbirds/train")
    load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="test").save_to_disk("./data/caltechbirds/test")
