# metrics_null_test.py
#
# Computes the null test experiment for the metrics computation strategy.


# Imports
import json
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from pprint import pprint
from warnings import warn

import torch
import torch_fidelity
from torch.utils.data import Subset
from torchvision.transforms import (
    CenterCrop,  # noqa: F401
    Compose,
    ConvertImageDtype,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,  # noqa: F401
)

from GaussianProxy.conf.training_conf import DataSet
from GaussianProxy.utils.data import BaseDataset

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from collections import defaultdict
from typing import Any

from GaussianProxy.utils.data import RandomRotationSquareSymmetry

# Disable grads globally
torch.set_grad_enabled(False)

# Precision
torch.set_float32_matmul_precision("highest")  # replaces torch.backends.cuda.matmul.allow_tf32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

############################################### Conf ##############################################
# Dataset
from my_conf.dataset.Jurkat_inference_hard_aug import dataset  # noqa: E402

assert dataset.dataset_params is not None

# now split the dataset into 2 non-overlapping parts, respecting classes proportions...
# ...and repeat that 10 times to get std of the metric
is_flip_or_rotation = lambda t: isinstance(
    t, (RandomHorizontalFlip | RandomVerticalFlip | RandomRotationSquareSymmetry)
)
flips_rot = [t for t in dataset.transforms.transforms if is_flip_or_rotation(t)]

# with or without augmentations:
# transforms = Compose(flips_rot + [ConvertImageDtype(torch.uint8)])
transforms = Compose(
    [
        ConvertImageDtype(torch.uint8),  # needed for FID computation
    ]
)


# replace the lambda sorting func in dataset_params with a normal function
def sorting_func(dataset_name: str, key: str | Path):
    if isinstance(key, Path):
        key = key.name
    match dataset_name:
        case "chromaLive6h_3ch_png_patches_380px_hard_aug":
            return -1 if key == "all_classes" else float(key.split("_")[-1])
        case "biotine_png_hard_aug":
            return -1 if key == "all_classes" else int(key)
        case "BBBC021_196_hard_aug_docetaxel":
            return -1 if key == "all_classes" else float(key)
        case "diabetic_retinopathy" | "diabetic_retinopathy_inference_hard_augmented":
            return -1 if key == "all_classes" else int(key)
        case "imagenet_n01917289_hard_aug_inference":
            int(key)
        case "Jurkat":
            phase_order = (
                "G1",
                "S",
                "G2",
                "Prophase",
                "Metaphase",
                "Anaphase",
                "Telophase",
            )
            return -1 if key == "all_classes" else phase_order.index(key)
        case _:
            raise ValueError(f"Unknown/Unimplemented dataset name: {dataset_name}")


dataset.dataset_params.sorting_func = partial(sorting_func, dataset.name)

nb_repeats = 10

batch_size = 512

metrics_save_path = Path(f"notebooks/evaluations/{dataset.name}/eval_metrics.json")
metrics_save_path.parent.mkdir(parents=True, exist_ok=True)

recompute = True

cuda_devices: tuple[int] = (2,)

############################################## Utils ##############################################


def process_subdirs(
    subdir: Path,
    dataset: DataSet,
    nb_elems_per_class: dict[str, int],
):
    assert dataset.dataset_params is not None
    this_class_elems = list(subdir.glob(f"*.{dataset.dataset_params.file_extension}"))
    nb_elems_per_class[subdir.name] = len(this_class_elems)
    random.shuffle(this_class_elems)
    ds1_elems = this_class_elems[: len(this_class_elems) // 2]
    ds2_elems = this_class_elems[len(this_class_elems) // 2 :]
    return ds1_elems, ds2_elems


def process_one_exp_rep(dataset: DataSet, subdirs: list[Path], nb_elems_per_class: dict[str, int]):
    assert dataset.dataset_params is not None
    ds1_elems = []
    ds2_elems = []

    with ThreadPoolExecutor() as subdirs_executor:
        futures = {
            subdir: subdirs_executor.submit(process_subdirs, subdir, dataset, nb_elems_per_class) for subdir in subdirs
        }
        for future in as_completed(futures.values()):
            ds1_elems_subdir, ds2_elems_subdir = future.result()
            ds1_elems.extend(ds1_elems_subdir)
            ds2_elems.extend(ds2_elems_subdir)

    assert abs(len(ds1_elems) - len(ds2_elems)) <= len(subdirs)
    print("Instantiating datasets for one experiment repeat...")
    ds1: BaseDataset = dataset.dataset_params.dataset_class(
        ds1_elems,
        transforms,
        dataset.expected_initial_data_range,
    )
    ds2: BaseDataset = dataset.dataset_params.dataset_class(
        ds2_elems,
        transforms,
        dataset.expected_initial_data_range,
    )
    return {"split1": ds1, "split2": ds2}


def compute_splits(dataset: DataSet):
    assert dataset.dataset_params is not None
    database_path = Path(dataset.path)
    print(f"Using dataset {dataset.name} from {database_path}")
    subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    subdirs.sort(key=dataset.dataset_params.sorting_func)

    exp_repeats: dict[str, dict[str, BaseDataset]] = {}
    nb_elems_per_class: dict[str, int] = {}
    print("Processing splits...")

    with ProcessPoolExecutor() as executor:
        futures = {}
        for exp_rep in range(nb_repeats):
            futures[executor.submit(process_one_exp_rep, dataset, subdirs, nb_elems_per_class)] = exp_rep

        for res in as_completed(futures):
            exp_rep = futures[res]
            exp_repeats[f"exp_rep_{exp_rep}"] = res.result()

    nb_elems_per_class["all_classes"] = sum(nb_elems_per_class.values())
    print("Experiment repeats:")
    for exp_rep, splits in exp_repeats.items():
        print(f"{exp_rep}: {len(splits['split1'])} vs {len(splits['split2'])} samples")

    return exp_repeats, nb_elems_per_class, subdirs


# FID
## Compute train vs train FIDs
def process_task_group_on_one_device(
    task_group: list[dict[str, Any]], cuda_device: int, temp_output_path: Path
) -> Path:
    """
    Worker function: set CUDA device, process each task and write results to a temporary JSON.
    Each task is a dict with keys:
      - exp_rep: experiment repeat key (str)
      - key: "all_classes" or class name (str)
      - split1, split2: datasets (BaseDataset)
      - batch_size: int
      - (if key != "all_classes") subdir: Path object
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} on  process {os.getpid()}")
    results: dict[str, dict[str, Any]] = defaultdict(dict)
    for task in task_group:
        exp_rep: str = task["exp_rep"]
        key: str = task["key"]
        split1: BaseDataset = task["split1"]
        split2: BaseDataset = task["split2"]
        batch_size: int = task["batch_size"]
        if key == "all_classes":
            print(f"Computing metrics {exp_rep} {key}: {len(split1)} vs {len(split2)} samples on device {cuda_device}")
            metrics = torch_fidelity.calculate_metrics(
                input1=split1,
                input2=split2,
                cuda=True,
                batch_size=batch_size,
                fid=True,
                prc=True,
                verbose=cuda_device == 0,
                samples_find_deep=True,
                cache=False,
            )
        else:
            subdir = task["subdir"]
            ds1_this_cl = Subset(
                split1,
                [i for i, e in enumerate(split1.samples) if e.parent == subdir],
            )
            ds2_this_cl = Subset(
                split2,
                [i for i, e in enumerate(split2.samples) if e.parent == subdir],
            )
            print(
                f"Computing metrics {exp_rep} {key}: {len(ds1_this_cl)} vs {len(ds2_this_cl)} samples on device {cuda_device}"
            )
            metrics = torch_fidelity.calculate_metrics(
                input1=ds1_this_cl,
                input2=ds2_this_cl,
                cuda=True,
                batch_size=batch_size,
                fid=True,
                prc=True,
                verbose=cuda_device == 0,
                cache=False,
            )
        results[exp_rep][key] = metrics
    with open(temp_output_path, "w") as f:
        json.dump(results, f)
    return temp_output_path


def compute_metrics(
    batch_size: int,
    metrics_save_path: Path,
    exp_repeats: dict[str, dict[str, BaseDataset]],
    subdirs: list[Path],
    cuda_devices: tuple[int],
):
    if metrics_save_path.exists():
        raise RuntimeError(f"File {metrics_save_path} already exists, not overwriting")

    # Build tasks: each task computes metrics for one key ("all_classes" or a specific class) in an experiment repeat.
    tasks: list[dict[str, Any]] = []
    for exp_rep, splits in exp_repeats.items():
        tasks.append(
            {
                "exp_rep": exp_rep,
                "key": "all_classes",
                "split1": splits["split1"],
                "split2": splits["split2"],
                "batch_size": batch_size,
            }
        )
        for subdir in subdirs:
            tasks.append(
                {
                    "exp_rep": exp_rep,
                    "key": subdir.name,
                    "split1": splits["split1"],
                    "split2": splits["split2"],
                    "batch_size": batch_size,
                    "subdir": subdir,
                }
            )

    print(f"Will process {len(tasks)} tasks in total")

    # Group tasks by cuda device (round-robin assignment)
    task_groups: dict[int, list[dict[str, Any]]] = {i: [] for i in cuda_devices}
    for idx, task in enumerate(tasks):
        device_id = cuda_devices[idx % len(cuda_devices)]
        task["cuda_device"] = device_id
        task_groups[device_id].append(task)

    tmp_files: dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=len(cuda_devices)) as executor:
        future_to_device: dict[Any, int] = {}
        for device_id, group in task_groups.items():
            tmp_file = metrics_save_path.parent / f"tmp_metrics_{device_id}.json"
            tmp_files[device_id] = tmp_file
            print(f"Processing {len(group)} tasks on device {device_id}, saving to {tmp_file}")
            future = executor.submit(process_task_group_on_one_device, group, device_id, tmp_file)
            future_to_device[future] = device_id
        for future in as_completed(future_to_device):
            future.result()  # Ensure completion

    # Merge temporary JSON files
    eval_metrics: dict[str, Any] = {}
    for tmp_file in tmp_files.values():
        with open(tmp_file) as f:
            partial = json.load(f)
        for exp_rep, metrics_dict in partial.items():
            if exp_rep not in eval_metrics:
                eval_metrics[exp_rep] = {}
            eval_metrics[exp_rep].update(metrics_dict)
        tmp_file.unlink()
        print(f"Deleted temporary file: {tmp_file}")

    # Save final merged metrics
    if not metrics_save_path.parent.exists():
        metrics_save_path.parent.mkdir(parents=True)
    with open(metrics_save_path, "w") as f:
        json.dump(eval_metrics, f)
    print(f"Saved metrics to {metrics_save_path}")
    return eval_metrics


############################################### Main ##############################################


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # for starting processes with CUDA

    print(f"Using dataset {dataset.name}")
    print(f"Using transforms:\n{transforms}")
    print(f"Will save metrics to {metrics_save_path}")
    print(f"Using devices {cuda_devices} for parallel execution")

    ### Compute splits
    exp_repeats, nb_elems_per_class, subdirs = compute_splits(dataset)

    ### Compute or load saved values
    if recompute:
        inpt = input("Confirm recompute (y/[n]):")
        if inpt != "y":
            warn(f"Will not recompute but load from {metrics_save_path}")
            try:
                with open(metrics_save_path) as f:
                    eval_metrics: dict[str, dict[str, dict[str, float]]] = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Metrics file {metrics_save_path} not found") from e
        else:
            warn("Will recompute using parallel metrics computation")
            eval_metrics = compute_metrics(batch_size, metrics_save_path, exp_repeats, subdirs, cuda_devices)
    else:
        warn(f"Will not recompute but load from {metrics_save_path}")
        with open(metrics_save_path) as f:
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
