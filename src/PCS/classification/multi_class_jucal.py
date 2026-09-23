import numpy as np
import os
import pickle
import copy
import math
from tqdm import tqdm

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder


# JUCAL Imports
from src.PCS.classification.calibration_utils import (
    JUCAL_calibration_oob,
    ensemble_JUCAL_calibration_oob,
    calibrate_then_pool_oob,
    ensemble_calibrate_then_pool_oob,
    JUCAL_calibration_deep,
    ensemble_JUCAL_calibration_deep,
    calibrate_then_pool_deep,
    ensemble_calibrate_then_pool_deep,
)
from src.metrics.classification_metrics import get_all_metrics
from src.PCS.classification.multi_class_pcs import MultiClassPCS


C1_COARSE = np.unique(np.append(np.linspace(0.3, 3.0, 50), 1.0))
C2_COARSE = np.unique(np.append(np.linspace(0.0, 10.0, 50), 1.0))
FINE_GRID_SIZE = 10


def make_splits(indices, y, fractions, seed=42):
    rng = np.random.default_rng(seed)
    # Shuffle each class once so every larger training subset contains the smaller ones
    # Also stratifying
    class_rows = [rng.permutation(indices[y[indices] == label]) for label in np.unique(y[indices])]

    subsets = {}
    for fraction in fractions:
        rows_per_class = [rows[: max(1, math.ceil(fraction * rows.size))] for rows in class_rows]
        subsets[fraction] = np.sort(np.concatenate(rows_per_class))

    for smaller, larger in zip(fractions, fractions[1:]):
        if not np.isin(subsets[smaller], subsets[larger]).all():
            raise AssertionError("Row conditions are not nested")
    return subsets


class MultiClassPCS_JUCAL(MultiClassPCS):
    def __init__(
        self,
        models,
        n_classes,
        num_bootstraps=100,
        seed=42,
        top_k=1,
        save_path=None,
        load_models=True,
        metric=log_loss,
        val_size=0.25,
        calibration_method="jucal",
    ):
        """
        MultiClassPCS_JUCAL

        Args:
            models: dictionary of model names and models
            num_bootstraps: number of bootstraps
            seed: random seed
            top_k: number of top models to use
            save_path: path to save the models
            load_models: whether to load the models from the save_path
            metric: metric to use for the prediction scores -- assume that higher is better
        """
        self.models = {
            model_name: copy.deepcopy(model) for model_name, model in models.items()
        }
        self.num_bootstraps = num_bootstraps
        self.seed = seed
        self.top_k = top_k
        self.save_path = save_path
        self.load_models = load_models
        self.metric = metric
        self.val_size = val_size
        self.n_classes = n_classes
        self.pred_scores = {model: np.inf for model in self.models}
        self.fill_val = None
        self.calibration_method = calibration_method

    def fit(self, X, y, fill=True):
        """
        Fit the models
        """
        le = LabelEncoder()
        y = le.fit_transform(y)
        self._label_encoder = le

        
        if fill:
            self.fill_val = 1/(2*len(y))

        train_inds = make_splits(np.array(range(len(y))), y, [1-self.val_size], seed=self.seed)[1-self.val_size]
        val_inds = np.setdiff1d(range(len(y)), train_inds)
        X_train = X[train_inds]
        X_calib = X[val_inds]
        y_train = y[train_inds]
        y_calib = y[val_inds]

        # X_train, X_calib, y_train, y_calib = train_test_split(
        #     X, y, test_size=self.val_size, random_state=self.seed, stratify=y
        # )

        assert len(np.unique(y_train)) == len(np.unique(y))

        self._train(
            X_train, y_train
        )  # train the models such that they are ready for calibration, saved in self.models
        self._pred_check(
            X_calib, y_calib
        )  # check the predictions of the models, saved in self.models
        self.top_k_models = self._get_top_k()
        self._train_top_k(X, y)
        self.best, self.cs = self.calibrate(X, y)

    def _train_top_k(self, X, y):
        """
        Train the models and store out-of-bag indices
        """
        # Initialize dictionaries once outside the model loop

        # self.bootstrap_models = {}
        self.oob_indices = {}
        self._flattened_bootstrap_models = []
        self._flattened_oob_indices = []
        self._classes_per_bootstrap = []

        for model_name, model in self.top_k_models.items():
            # Initialize lists for each model
            # self.bootstrap_models[model_name] = []
            self.oob_indices[model_name] = []

            for i in tqdm(
                range(self.num_bootstraps), desc=f"Training {model_name} models"
            ):
                bootstrap_seed = self.seed + i
                # Try to load existing bootstrap model and OOB indices if enabled
                model_path = (
                    f"{self.save_path}/pcs_jucal/models/{model_name}_model_seed_{bootstrap_seed}.pkl"
                    if self.save_path
                    else None
                )
                oob_path = (
                    f"{self.save_path}/pcs_jucal/oob_indices/{model_name}_oob_seed_{bootstrap_seed}.pkl"
                    if self.save_path
                    else None
                )
                classes_path = (
                    f"{self.save_path}/pcs_jucal/classes/{model_name}_classes_seed_{bootstrap_seed}.pkl"
                    if self.save_path
                    else None
                )
                bootstrap_model = None

                if (
                    self.load_models
                    and model_path
                    and os.path.exists(model_path)
                    and os.path.exists(oob_path)
                    and os.path.exists(classes_path)
                ):
                    
                    with open(model_path, "rb") as f:
                        bootstrap_model = pickle.load(f)
                    with open(oob_path, "rb") as f:
                        oob_indices = pickle.load(f)
                    with open(classes_path, "rb") as f:
                        unique_classes = pickle.load(f)
                    self.oob_indices[model_name].append(oob_indices)
                    self._flattened_oob_indices.append(oob_indices)
                    self._classes_per_bootstrap.append(unique_classes)
                else:
                    # Bootstrap the data
                    n_samples = len(X)
                    class_counts = np.bincount(y.astype(int))
                    class_idx_to_freq = {
                        i: class_counts[i] / n_samples for i in range(len(class_counts))
                    }
                    weights = np.ones(n_samples)
                    # for i in range(n_samples):
                    #     weights[i] = class_idx_to_freq[y[i]]
                    # weights = weights / weights.sum()
                    weights = weights / weights.sum()

                    # bootstrap_indices = resample(
                    #     range(n_samples), n_samples=n_samples, replace=True, random_state=bootstrap_seed, stratify=y
                    # )

                    rng = np.random.default_rng(bootstrap_seed)
                    bootstrap_indices = rng.choice(
                        range(n_samples), size=n_samples, replace=True, p=weights
                    )
                    oob_indices = list(set(range(n_samples)) - set(bootstrap_indices))



                    X_boot = X[bootstrap_indices]
                    y_boot_ = y[bootstrap_indices]

                    unique_classes = np.unique(y_boot_)

                    self._classes_per_bootstrap.append(unique_classes)

                    # New label encodings in case y_boot_ does not include certain classes

                    leb = LabelEncoder()
                    y_boot = leb.fit_transform(y_boot_)

                    # Store OOB indices
                    self.oob_indices[model_name].append(oob_indices)
                    self._flattened_oob_indices.append(oob_indices)
                    # Create and fit bootstrap model
                    bootstrap_model = clone(model)
                    bootstrap_model.fit(X_boot, y_boot)

                    # Save the bootstrap model and OOB indices if save path is provided
                    if model_path:
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, "wb") as f:
                            pickle.dump(bootstrap_model, f)
                        with open(oob_path, "wb") as f:
                            pickle.dump(oob_indices, f)
                        with open(classes_path, "wb") as f:
                            pickle.dump(unique_classes, f)

                # self.bootstrap_models[model_name].append(bootstrap_model)
                self._flattened_bootstrap_models.append(bootstrap_model)
            print(f"Finished training {model_name} models")

    def calibrate(self, X, y):
        # JUCAL calibration

        calib_path = (
            f"{self.save_path}/pcs_jucal/{self.calibration_method}/calibrations.pkl"
            if self.save_path
            else None
        )

        if (
            self.load_models
            and os.path.exists(calib_path)
        ):
            with open(calib_path, "rb") as f:
                best_NLL, best_cs = pickle.load(f)

        else:
            if self.calibration_method == "jucal":
                best_NLL, best_cs = JUCAL_calibration_oob(
                    X=X,
                    y=y,
                    oob_indices=self._flattened_oob_indices,
                    bootstrap_models=self._flattened_bootstrap_models,
                    n_classes=self.n_classes,
                    classes_per_bootstrap=self._classes_per_bootstrap,
                    metric=self.metric,
                    C1=C1_COARSE,
                    C2=C2_COARSE,
                    K=FINE_GRID_SIZE,
                    fill_val=self.fill_val,
                )

            elif self.calibration_method == "ctp":
                best_NLL, best_cs = calibrate_then_pool_oob(
                    X=X,
                    y=y,
                    oob_indices=self._flattened_oob_indices,
                    bootstrap_models=self._flattened_bootstrap_models,
                    n_classes=self.n_classes,
                    classes_per_bootstrap=self._classes_per_bootstrap,
                    metric=self.metric,
                    C1=C1_COARSE,
                    K=FINE_GRID_SIZE,
                    fill_val=self.fill_val,
                )

            if calib_path:
                os.makedirs(os.path.dirname(calib_path), exist_ok=True)
                with open(calib_path, "wb") as f:
                    pickle.dump((best_NLL, best_cs), f)

        return best_NLL, best_cs

    def ensemble(self, X):
        # Explicitly output the ensemble (n_samples, n_classes, n_models)

        if self.calibration_method == "jucal":
            return ensemble_JUCAL_calibration_oob(
                X=X,
                bootstrap_models=self._flattened_bootstrap_models,
                c1=self.cs[0],
                c2=self.cs[1],
                n_classes=self.n_classes,
                classes_per_bootstrap=self._classes_per_bootstrap,
                fill_val=self.fill_val,
            )


        elif self.calibration_method == "ctp":
            return ensemble_calibrate_then_pool_oob(
                X=X,
                bootstrap_models=self._flattened_bootstrap_models,
                c1=self.cs,
                n_classes=self.n_classes,
                classes_per_bootstrap=self._classes_per_bootstrap,
                fill_val=self.fill_val,
            )


    def predict(self, X):
        return np.nanmean(self.ensemble(X), axis=2)


import torch
from experiments.scripts.train_models import create_model


class MultiClassPCS_JUCAL_DEEP(MultiClassPCS):
    """
    We are assuming a pretrained model.  
    """
    def __init__(
        self,
        model_path,
        num_classes,
        ensemble_size=100,
        seed=42,
        metric=log_loss,
        calibration_method="jucal",
    ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # model_dict = torch.load(model_path)
        # self.model = create_model(num_classes).to(self.device)
        # self.model.load_state_dict(model_dict["state_dict"])
        # self.model.eval()

        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.metric = metric
        self.calibration_method = calibration_method


    def create_ensemble(self, model, method):
        """
        Create ensemble for pretrained ResNet18 model
        """

        self.models = []
        model.eval()
        g = torch.Generator()

        if method == "perturb":
            # These don't change the batch norm layers (initialization is deterministic so variance is zero)

            for i in range(self.ensemble_size-1):
                state_dict = model.state_dict().copy()
                for i, (name, param) in enumerate(state_dict.items()):
                    if ("conv" in name) or ("downsample.0" in name):
                        g.manual_seed(self.seed+i)
                        new_param = param + torch.normal(0, 2/(param.size()[0]*param.size()[2]**2), size=param.size(), generator=g)
                    elif "fc" in name:
                        g.manual_seed(self.seed+i)
                        new_param = param + torch.normal(0, 1/(3*512), size=param.size(), generator=g)
                    else:
                        new_param = param
                    state_dict[name] = new_param

            new_model = copy.deepcopy(model)
            new_model.load_state_dict(state_dict)

            new_model.eval()

            self.models.append(new_model)
                
        elif method == "dropout":
            # These can change the batchnorm layers (because they are trainable)

            for i in range(self.ensemble_size-1):
                state_dict = model.state_dict().copy()
                for i, (name, param) in enumerate(model.parameters()):
                    if param.requires_grad:
                        g.manual_seed(self.seed+i)
                        drop_probs = torch.abs(param)/torch.sum(torch.abs(param))
                        new_param = (torch.rand(size=param.size(), generator=g) >= drop_probs).long()*param
                    else:
                        new_param = param
                    state_dict[name] = new_param

            new_model = copy.deepcopy(model)
            new_model.load_state_dict(state_dict)

            new_model.eval()

            self.models.append(new_model)


    def calibrate(self, X, y):
        # JUCAL calibration

        calib_path = (
            f"{self.save_path}/jucal/{self.calibration_method}/calibrations.pkl"
            if self.save_path
            else None
        )

        if (
            self.load_models
            and os.path.exists(calib_path)
        ):
            with open(calib_path, "rb") as f:
                best_NLL, best_cs = pickle.load(f)

        else:
            if self.calibration_method == "jucal":
                best_NLL, best_cs = JUCAL_calibration_deep(
                    X=X,
                    y=y,
                    models=self.models,
                    n_classes=self.n_classes,
                    metric=self.metric,
                    C1=C1_COARSE,
                    C2=C2_COARSE,
                    K=FINE_GRID_SIZE,
                )

            elif self.calibration_method == "ctp":
                best_NLL, best_cs = calibrate_then_pool_deep(
                    X=X,
                    y=y,
                    models=self.models,
                    n_classes=self.n_classes,
                    metric=self.metric,
                    C1=C1_COARSE,
                    K=FINE_GRID_SIZE,
                )

            if calib_path:
                os.makedirs(os.path.dirname(calib_path), exist_ok=True)
                with open(calib_path, "wb") as f:
                    pickle.dump((best_NLL, best_cs), f)

        return best_NLL, best_cs

    def ensemble(self, X):
        # Explicitly output the ensemble (n_samples, n_classes, n_models)

        if self.calibration_method == "jucal":
            return ensemble_JUCAL_calibration_deep(
                X=X,
                mdoels=self.models,
                c1=self.cs[0],
                c2=self.cs[1],
            )


        elif self.calibration_method == "ctp":
            return ensemble_calibrate_then_pool_oob(
                X=X,
                bootstrap_models=self.models,
                c1=self.cs,
                )

    def predict(self, X):
        return np.nanmean(self.ensemble(X), axis=2)
        
    


        



if __name__ == "__main__":
    models = {
        "rf": RandomForestClassifier(
            n_estimators=5, random_state=42, min_samples_leaf=50
        ),
        "lr": LogisticRegression(random_state=42),
    }
    X, y = make_classification(
        n_samples=250, n_features=10, n_classes=10, n_informative=5
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pcs_JUCAL = MultiClassPCS_JUCAL(
        models,
        n_classes=len(np.unique(y)),
        num_bootstraps=500,
        seed=42,
        top_k=1,
        save_path="./models",
        load_models=False,
        metric=log_loss
    )
    pcs_JUCAL.fit(X_train, y_train)
    probs = pcs_JUCAL.predict(X_test)