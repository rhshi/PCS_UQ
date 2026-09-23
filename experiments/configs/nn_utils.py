import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop, to_pil_image
from PIL import Image
from datasets import load_from_disk

import time
import numpy as np
import random

CIFAR_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])

BIRDS_NORMALIZE = transforms.Normalize(mean=[0.4635, 0.4746, 0.4108],
  std=[0.2377, 0.2348, 0.2638])

TINY_NORMALIZE = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
  std=[0.2764, 0.2689, 0.2816])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):

        self.data = load_from_disk(path).with_format("numpy")
        self.transform = transform

        self.in_str = "img"
        self.out_str = "fine_label"

        self.labels = []
        for i in range(len(self.data)):
          self.labels.append(self.data[i][self.out_str])

        self.num_classes = len(np.unique(self.labels))

    def __getitem__(self, idx):
        sample = self.data[idx][self.in_str]
        target = self.data[idx][self.out_str]
        if self.transform:
          sample = self.transform(sample.copy())
        return sample, target 

    def __len__(self):
        return len(self.data)


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.data = load_from_disk(path).with_format("numpy")
        self.transform = transform

        self.in_str = "image"
        self.out_str = "label"

        self.labels = []
        for i in range(len(self.data)):
          self.labels.append(self.data[i][self.out_str])

        self.num_classes = len(np.unique(self.labels))

    def __getitem__(self, idx):
        sample = self.data[idx][self.in_str]
        if len(sample.shape) == 2:
          sample = np.stack((sample,) * 3, axis=-1)

        target = self.data[idx][self.out_str]
        if self.transform:
          sample = self.transform(sample.copy())
        return sample, target 

    def __len__(self):
        return len(self.data)

class CaltechBirds(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.data = load_from_disk(path).with_format("numpy")
        self.transform = transform

        self.in_str = "image"
        self.out_str = "label"

        self.labels = []
        for i in range(len(self.data)):
          self.labels.append(self.data[i][self.out_str])

        self.num_classes = len(np.unique(self.labels))

    def __getitem__(self, idx):
        sample = self.data[idx][self.in_str]
        if len(sample.shape) == 2:
          sample = np.stack((sample,) * 3, axis=-1)
        target = self.data[idx][self.out_str]
        sample = np.array(center_crop(to_pil_image(sample), np.max(sample.shape[:1])).resize((64, 64), resample=Image.BICUBIC))

        if self.transform:
          sample = self.transform(sample.copy())
        return sample, target 

    def __len__(self):
        return len(self.data)


def get_transformations(dataset):
  if dataset == "CIFAR100":
    size = 32
    normalize = CIFAR_NORMALIZE
  elif dataset == "TinyImageNet":
    size = 64
    normalize = TINY_NORMALIZE
  elif dataset == "CaltechBirds":
    size = 64
    normalize = BIRDS_NORMALIZE

  train_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
    ])

  test_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
    ])

  return train_transformations, test_transformations
  

# def get_pretrained_model(model_str, num_classes):
#   if model_str == "resnet":
#     model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
#   elif model_str == "efficientnet":
#     model = timm.create_model("hf_hub:timm/tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=num_classes)
#   elif model_str == "densenet":
#     model = timm.create_model("hf_hub:timm/densenet121.tv_in1k", pretrained=True, num_classes=num_classes)
  
#   return model

def get_dataloaders(train_data, val_data, batch_size=128, seed=42):

  g = torch.Generator()
  g.manual_seed(seed)

  train_loader = torch.utils.data.DataLoader(
          train_data,
          batch_size=batch_size,
          shuffle=True,
          num_workers=0,
          generator=g,
          )


  val_loader = torch.utils.data.DataLoader(
          val_data,
          batch_size=batch_size,
          shuffle=False,
          num_workers=0
          )

  return train_loader, val_loader


def train_epoch(train_loader, model, criterion, optimizer, epoch, print_freq=50, shuffle_seed=42):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # set shuffle seeds

    random.seed(shuffle_seed+epoch+1)
    torch.manual_seed(shuffle_seed+epoch+1)
    torch.cuda.manual_seed(shuffle_seed+epoch+1)
    np.random.seed(shuffle_seed+epoch+1)


    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.to(device)
        input_var = input.to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, print_freq=50):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target_var = target.to(device)
            input_var = input.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res