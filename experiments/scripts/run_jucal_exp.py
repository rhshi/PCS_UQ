
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from src.PCS.classification.multi_class_jucal import MultiClassPCS_JUCAL, make_splits
from experiments.configs.classification_configs import get_classification_datasets

from experiments.configs.classification_consts import MODELS, DATASETS
SEEDS = [42, 1042, 2042, 3042, 4042]
SAMPLE_PROPORTIONS = [0.0625, 0.125, 0.25, 0.5, 1]

for dataset_key in range(len(DATASETS)):
    print(DATASETS[dataset_key])

    X, y, _, importance = get_classification_datasets(DATASETS[dataset_key])
    num_classes = len(np.unique(y))

    for seed in SEEDS:
        print(seed)
        subsets_ = make_splits(np.array(range(len(y))), y, [0.75], seed=3*seed)
        train_inds = subsets_[0.75]
        test_inds = np.setdiff1d(range(len(y)), train_inds)

        ytest = y[test_inds]

        assert len(np.unique(y)) == len(np.unique(ytest))

        subsets = make_splits(train_inds, y, SAMPLE_PROPORTIONS, seed=2*seed)

        for (i, sample_proportion) in enumerate(SAMPLE_PROPORTIONS):
            for j in range(len(importance)):

                print(sample_proportion, j+1)
                save_path = f"./models/{DATASETS[dataset_key]}/seed_{seed}/fill/sample_proportion_{sample_proportion}_num_features_{j+1}"

                Xtrain = (X[importance.iloc[0:j+1]["feature"]].to_numpy())[subsets[sample_proportion]]
                Xtest = (X[importance.iloc[0:j+1]["feature"]].to_numpy())[test_inds]

                assert len(np.unique(y)) == len(np.unique(y[subsets[sample_proportion]]))
                assert not (set(test_inds) & set(subsets[sample_proportion]))
                

                # JUCAL

                # print("JUCAL")

                # pcs_JUCAL = MultiClassPCS_JUCAL(
                #     MODELS,
                #     num_bootstraps=500,
                #     n_classes=len(np.unique(y)),
                #     seed=seed,
                #     top_k=2,
                #     save_path=save_path,
                #     load_models=True,
                #     metric=log_loss,
                #     calibration_method="jucal",
                # )
                # pcs_JUCAL.fit(Xtrain, y[subsets[sample_proportion]], fill=True)

                # Calibrate then pool

                print("Calibrate then pool")

                pcs_ctp = MultiClassPCS_JUCAL(
                    MODELS,
                    num_bootstraps=500,
                    n_classes=len(np.unique(y)),
                    seed=seed,
                    top_k=2,
                    save_path=save_path,
                    load_models=True,
                    metric=log_loss,
                    calibration_method="ctp",
                )
                pcs_ctp.fit(Xtrain, y[subsets[sample_proportion]], fill=True)
                print("==========================================================")