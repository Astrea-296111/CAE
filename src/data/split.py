import numpy as np


def split_indices(size, num_train, num_val, seed=42):
    if min(num_train, num_val) < 1 or num_train + num_val >= size:
        raise ValueError("Train, validation and test must all be nonempty")
    order = np.random.default_rng(seed).permutation(size)
    return {"train": order[:num_train], "val": order[num_train:num_train + num_val],
            "test": order[num_train + num_val:]}


def validate_thickness_ood(train_range, ood_range):
    train_low, train_high = train_range
    ood_low, ood_high = ood_range
    if not 0 < train_low < train_high < ood_low < ood_high:
        raise ValueError("Thickness OOD must lie strictly above the training range")
