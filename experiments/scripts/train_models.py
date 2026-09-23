import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import os
import argparse

from experiments.configs.nn_utils import CIFAR100, TinyImageNet, CaltechBirds, get_transformations, get_dataloaders, train_epoch, validate, save_checkpoint


DATASETS = ["CIFAR100", "TinyImageNet", "CaltechBirds"]


def train_val_split(train_, split_size=0.2, seed=43):
    train_inds, val_inds = train_test_split(
        range(len(train_)),
        test_size=split_size,
        stratify=train_.labels,
        random_state=seed
    )

    train_data = Subset(train_, train_inds)
    val_data = Subset(train_, val_inds)

    return train_data, val_data

def create_model(num_classes):
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    return model

def create_train_utils(model, lr=0.1, momentum=0.9, weight_decay=1e-4, T_max=200):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    return criterion, optimizer, lr_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10, help="Save model interval")
    parser.add_argument("--seed", type=int, default=3821, help="Seed")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for DATASET in DATASETS:
        save_dir = f"./experiments/models/{DATASET}/"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        train_transforms, test_transforms = get_transformations(DATASET)

        if DATASET == "CIFAR100":
            train_ = CIFAR100("./data/cifar100/train", transform=train_transforms)
            test_data = CIFAR100("./data/cifar100/test", transform=test_transforms)

        elif DATASET == "TinyImageNet":
            train_ = TinyImageNet("./data/tinyimagenet/train", transform=train_transforms)
            test_data = TinyImageNet("./data/tinyimagenet/test", transform=test_transforms)

        elif DATASET == "CaltechBirds":
            train_ = CaltechBirds("./data/caltechbirds/train", transform=train_transforms)
            test_data = CaltechBirds("./data/caltechbirds/test", transform=test_transforms)

        data_seed = 2*args.seed

        train_data, val_data = train_val_split(train_, seed=data_seed)
        model = create_model(train_.num_classes).to(device)

        shuffle_seed = 3*args.seed

        train_loader, val_loader = get_dataloaders(train_data, val_data, seed=shuffle_seed)
        criterion, optimizer, lr_scheduler = create_train_utils(model)

        for epoch in range(1, args.epochs+1):
            train_epoch(train_loader, model, criterion, optimizer, epoch, shuffle_seed=shuffle_seed)
            prec1 = validate(val_loader, model, criterion)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict(),
                }, filename=os.path.join(save_dir, f"epoch_{epoch}_checkpoint.pt"))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(save_dir, 'model.pt'))