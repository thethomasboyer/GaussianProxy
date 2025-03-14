"""
Script used to test the hypothesis that augmentations
can mimic different data splits.

- take one single split on the dataset (vs 10 in the "evaluations" notebook)
- compute 10 times the FID between the two splits, with random augmentations each time
- compare obtained FIDs to the ones obtained in the "evaluations" notebook
"""

import json
import random
import sys
from pathlib import Path
from pprint import pprint
from warnings import warn

import seaborn as sns
import torch
import torch_fidelity
from torch.utils.data import Subset
from torchvision.transforms import Compose, ConvertImageDtype, RandomHorizontalFlip, RandomVerticalFlip
from tqdm.notebook import trange

sys.path.insert(0, "..")
from GaussianProxy.utils.data import RandomRotationSquareSymmetry

torch.set_grad_enabled(False)
sns.set_theme(context="paper")

# Dataset
from my_conf.dataset.BBBC021_196_docetaxel_inference import BBBC021_196_docetaxel_inference as dataset  # noqa: E402

assert dataset.dataset_params is not None
database_path = Path(dataset.path)
print(f"Using dataset {dataset.name} from {database_path}")
subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
subdirs.sort(key=dataset.dataset_params.sorting_func)
print(f"Found {len(subdirs)} classes: {', '.join(e.name for e in subdirs)}")

# now split the dataset into 2 non-overlapping parts, respecting classes proportions...
is_flip_or_rotation = lambda t: isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry))
flips_rot = [t for t in dataset.transforms.transforms if is_flip_or_rotation(t)]
transforms = Compose(flips_rot + [ConvertImageDtype(torch.uint8)])
print(f"Using transforms:\n{transforms}")
nb_repeats = 10
nb_elems_per_class: dict[str, int] = {}

# create split and datasets once
ds1_elems = []
ds2_elems = []
for subdir in subdirs:
    this_class_elems = list(subdir.glob(f"*.{dataset.dataset_params.file_extension}"))
    nb_elems_per_class[subdir.name] = len(this_class_elems)
    random.shuffle(this_class_elems)
    new_ds1_elems = this_class_elems[: len(this_class_elems) // 2]
    new_ds2_elems = this_class_elems[len(this_class_elems) // 2 :]
    ds1_elems += new_ds1_elems
    ds2_elems += new_ds2_elems
    assert len(new_ds1_elems) + len(new_ds2_elems) == len(this_class_elems), (
        f"{len(new_ds1_elems)} + {len(new_ds2_elems)} != {len(this_class_elems)}"
    )
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
print("ds1:", ds1)
print("ds2:", ds2)

nb_elems_per_class["all_classes"] = sum(nb_elems_per_class.values())
print("nb_elems_per_class:", nb_elems_per_class)


# FID
# ## Compute train vs train FIDs
def compute_metrics(batch_size: int, metrics_save_path: Path):
    eval_metrics = {}

    for exp_rep in trange(nb_repeats, unit="experiment repeat"):
        metrics_dict: dict[str, dict[str, float]] = {}
        metrics_dict["all_classes"] = torch_fidelity.calculate_metrics(
            input1=ds1,
            input2=ds2,
            cuda=True,
            batch_size=batch_size,
            isc=True,
            fid=True,
            prc=True,
            verbose=True,
            samples_find_deep=True,
        )
        # per-class
        for subdir in subdirs:
            this_class_ds1_idxes = [i for i, e in enumerate(ds1_elems) if e.parent == subdir]
            this_class_ds2_idxes = [i for i, e in enumerate(ds2_elems) if e.parent == subdir]
            ds1_this_cl = Subset(ds1, this_class_ds1_idxes)
            ds2_this_cl = Subset(ds2, this_class_ds2_idxes)
            assert abs(len(ds1_this_cl) - len(ds2_this_cl)) <= 1
            assert len(ds1_this_cl) + len(ds2_this_cl) == nb_elems_per_class[subdir.name], (
                f"{len(ds1_this_cl)} + {len(ds2_this_cl)} != {nb_elems_per_class[subdir.name]}"
            )
            metrics_dict_cl = torch_fidelity.calculate_metrics(
                input1=ds1_this_cl,
                input2=ds2_this_cl,
                cuda=True,
                batch_size=batch_size,
                isc=True,
                fid=True,
                prc=True,
                verbose=True,
            )
            metrics_dict[subdir.name] = metrics_dict_cl

        eval_metrics[exp_rep] = metrics_dict

    if metrics_save_path.exists():
        raise RuntimeError(f"File {metrics_save_path} already exists, not overwriting")
    if not metrics_save_path.parent.exists():
        metrics_save_path.parent.mkdir(parents=True)
    with open(metrics_save_path, "w") as f:
        json.dump(eval_metrics, f)

    return eval_metrics


batch_size = 512
metrics_save_path = Path(f"evaluations/{dataset.name}/eval_metrics_TEST_REPS_WITH_AUGS.json")
recompute = True

if recompute:
    inpt = input("Confirm recompute (y/[n]):")
    if inpt != "y":
        warn(f"Will not recompute but load from {metrics_save_path}")
        with open(metrics_save_path) as f:
            eval_metrics = json.load(f)
    else:
        warn("Will recompute")
        assert not metrics_save_path.exists(), f"Refusing to overwrite {metrics_save_path}"
        eval_metrics = compute_metrics(batch_size, metrics_save_path)
else:
    warn(f"Will not recompute but load from {metrics_save_path}")
    with open(metrics_save_path) as f:
        eval_metrics = json.load(f)

pprint(eval_metrics)

# Extract class names and FID scores for training data vs training data
class_names = list(eval_metrics[0].keys())
fid_scores_by_class_train = {class_name: [] for class_name in class_names}

for exp_rep in eval_metrics.values():
    for class_name in class_names:
        fid_scores_by_class_train[class_name].append(exp_rep[class_name]["frechet_inception_distance"])

pprint(fid_scores_by_class_train)
