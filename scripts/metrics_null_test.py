# metrics_null_test.py
#
# Computes the null test experiment for the metrics computation strategy.


# Imports
import json
import os
import random
import sys
from pathlib import Path
from pprint import pprint
from warnings import warn

import torch
import torch_fidelity
from torch.utils.data import Subset
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from tqdm import tqdm, trange

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from GaussianProxy.utils.data import RandomRotationSquareSymmetry

# Disable grads globally
torch.set_grad_enabled(False)

# Dataset
from my_conf.dataset.chromalive6h_3ch_png_hard_aug_inference import dataset  # noqa: E402

assert dataset.dataset_params is not None
database_path = Path(dataset.path)
print(f"Using dataset {dataset.name} from {database_path}")
subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
subdirs.sort(key=dataset.dataset_params.sorting_func)

# now split the dataset into 2 non-overlapping parts, respecting classes proportions...
# ...and repeat that 10 times to get std of the metric
is_flip_or_rotation = lambda t: isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry))
flips_rot = [t for t in dataset.transforms.transforms if is_flip_or_rotation(t)]

# with or without augmentations:
# transforms = Compose(flips_rot + [ConvertImageDtype(torch.uint8)])
transforms = Compose([ConvertImageDtype(torch.uint8)])

print(f"Using transforms:\n{transforms}")
nb_repeats = 10
exp_repeats = {}
nb_elems_per_class = {}

for exp_rep in trange(nb_repeats, desc="Building splits of experiment repeats", unit="repeat"):
    ds1_elems = []
    ds2_elems = []
    for subdir in subdirs:
        this_class_elems = list(subdir.glob(f"*.{dataset.dataset_params.file_extension}"))
        nb_elems_per_class[subdir.name] = len(this_class_elems)
        random.shuffle(this_class_elems)
        ds1_elems += this_class_elems[: len(this_class_elems) // 2]
        ds2_elems += this_class_elems[len(this_class_elems) // 2 :]

    assert abs(len(ds1_elems) - len(ds2_elems)) <= len(subdirs)
    ds1 = dataset.dataset_params.dataset_class(
        ds1_elems,
        transforms,
        dataset.expected_initial_data_range,
    )
    ds2 = dataset.dataset_params.dataset_class(
        ds2_elems,
        transforms,
        dataset.expected_initial_data_range,
    )
    exp_repeats[f"exp_rep_{exp_rep}"] = {"split1": ds1, "split2": ds2}

nb_elems_per_class["all_classes"] = sum(nb_elems_per_class.values())
print("Experiment repeats:")
pprint({k: {inner_k: str(inner_v)} for k, v in exp_repeats.items() for inner_k, inner_v in v.items()})


# FID
## Compute train vs train FIDs
def compute_metrics(batch_size: int, metrics_save_path: Path):
    eval_metrics = {}

    for exp_rep in tqdm(exp_repeats, unit="experiment repeat", desc="Computing metrics"):
        metrics_dict: dict[str, dict[str, float]] = {}
        if exp_rep == "exp_rep_0":
            print(
                f"All classes: {len(exp_repeats[exp_rep]['split1'])} vs {len(exp_repeats[exp_rep]['split2'])} samples"
            )
        metrics_dict["all_classes"] = torch_fidelity.calculate_metrics(
            input1=exp_repeats[exp_rep]["split1"],
            input2=exp_repeats[exp_rep]["split2"],
            cuda=True,
            batch_size=batch_size,
            isc=False,
            fid=True,
            prc=False,
            verbose=True,
            samples_find_deep=True,
        )
        # per-class
        for subdir in subdirs:
            ds1_this_cl = Subset(
                exp_repeats[exp_rep]["split1"],
                [i for i, e in enumerate(ds1_elems) if e.parent == subdir],
            )
            ds2_this_cl = Subset(
                exp_repeats[exp_rep]["split2"],
                [i for i, e in enumerate(ds2_elems) if e.parent == subdir],
            )
            if exp_rep == "exp_rep_0":
                print(f"Will use {len(ds1_this_cl)} and {len(ds2_this_cl)} elements for splits of class {subdir.name}")
            assert abs(len(ds1_this_cl) - len(ds2_this_cl)) <= 1
            assert len(ds1_this_cl) + len(ds2_this_cl) == nb_elems_per_class[subdir.name]
            metrics_dict_cl = torch_fidelity.calculate_metrics(
                input1=ds1_this_cl,
                input2=ds2_this_cl,
                cuda=True,
                batch_size=batch_size,
                isc=False,
                fid=True,
                prc=False,
                verbose=True,
            )
            metrics_dict[subdir.name] = metrics_dict_cl
        eval_metrics[exp_rep] = metrics_dict  # for saving to json

    if metrics_save_path.exists():
        raise RuntimeError(f"File {metrics_save_path} already exists, not overwriting")
    if not metrics_save_path.parent.exists():
        metrics_save_path.parent.mkdir(parents=True)
    with open(metrics_save_path, "w") as f:
        json.dump(eval_metrics, f)
    print(f"Saved metrics to {metrics_save_path}")

    return eval_metrics


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
batch_size = 512
metrics_save_path = Path(f"notebooks/evaluations/{dataset.name}/eval_metrics.json")
print(f"Will save metrics to {metrics_save_path}")
recompute = True

### Compute or load saved values
if recompute:
    inpt = input("Confirm recompute (y/[n]):")
    if inpt != "y":
        warn(f"Will not recompute but load from {metrics_save_path}")
        with open(metrics_save_path, "r") as f:
            eval_metrics: dict[str, dict[str, dict[str, float]]] = json.load(f)
    else:
        warn("Will recompute")
        eval_metrics = compute_metrics(batch_size, metrics_save_path)
else:
    warn(f"Will not recompute but load from {metrics_save_path}")
    with open(metrics_save_path, "r") as f:
        eval_metrics: dict[str, dict[str, dict[str, float]]] = json.load(f)
print("Metrics computed:")
pprint(eval_metrics)
# Extract class names and FID scores for training data vs training data
class_names = list(eval_metrics["exp_rep_0"].keys())
fid_scores_by_class_train = {class_name: [] for class_name in class_names}

for exp_rep in eval_metrics.values():
    for class_name in class_names:
        fid_scores_by_class_train[class_name].append(exp_rep[class_name]["frechet_inception_distance"])

print("FID scores by timesteps:")
pprint(fid_scores_by_class_train)
