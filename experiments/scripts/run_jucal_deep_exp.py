from src.PCS.classification.multi_class_jucal import MultiClassPCS_JUCAL_DEEP
from experiments.scripts.train_models import train_val_split, create_model
from experiments.configs.nn_utils import CIFAR100, TinyImageNet, CaltechBirds, get_transformations

import os
import torch
from sklearn.model_selection import train_test_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASETS = ["CIFAR100", "TinyImageNet", "CaltechBirds"]

SEED = 3821 # MAKE SURE THE SEED IS THE SAME AS THE train_models.py SEED; THIS IS IMPORTANT FOR GETTING THE SAME VALIDATION SET
DATA_SEED = 2*SEED # THIS SEED IS USED TO OBTAIN TRAIN/VAL SPLIT
SHUFFLE_SEED = 3*SEED # THIS SEED WILL BE USED TO CREATE THE ENSEMBLE

EPOCHS = 200 # MAKE SURE THIS MATCHES train_models.py

DATASET = "CIFAR100"

save_dir = f"./experiments/models/{DATASET}/" # MAKE SURE THIS MATCHES save_dir IN train_models.py

###################################################################################################################################

_, test_transforms = get_transformations(DATASET)

if DATASET == "CIFAR100":
    train_ = CIFAR100("./data/cifar100/train", transform=test_transforms)

elif DATASET == "TinyImageNet":
    train_ = TinyImageNet("./data/tinyimagenet/train", transform=test_transforms)

elif DATASET == "CaltechBirds":
    train_ = CaltechBirds("./data/caltechbirds/train", transform=test_transforms)

train_inds, val_inds = train_test_split(
    range(len(train_)),
    test_size=0.25,
    stratify=train_.labels,
    random_state=DATA_SEED
)

Xval = train_.samples[val_inds]
yval = train_.labels[val_inds]

model = create_model(train_.num_classes).to(device)
model_dict = torch.load(os.path.join(save_dir, f"epoch_{EPOCHS}_checkpoint.pt"))
model.load_state_dict(model_dict["state_dict"]).to(device)
model.eval()

JUCAL_DEEP = MultiClassPCS_JUCAL_DEEP(
    model=model,
    num_classes=train_.num_classes, 
    ensemble_size=100,
    seed=SHUFFLE_SEED,
    save_path=save_dir,
    method="perturb",
)

# JUCAL_DEEP.create_ensemble(model)

# create_ensemble=False does not create and save all models, but creates each model from deterministic procedure based
# on seed, gets logits, and discards
JUCAL_DEEP.calibrate(Xval, yval, create_ensemble=False)
