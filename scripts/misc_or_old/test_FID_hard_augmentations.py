"""
Script used to test the hypothesis that augmentations
can mimic different data splits.

- take one single split on the dataset (vs 10 in the "evaluations" notebook)
- multiply 8 times each sample with the 8 square symmetries
- compute the FID between the two augmented splits
- save it in a json file
"""

import concurrent.futures
import json
import random
import sys
from pathlib import Path
from pprint import pprint
from warnings import warn

import seaborn as sns
import torch
import torch_fidelity
from PIL import Image
from rich.traceback import install
from torch import Tensor
from torchvision.transforms import Compose, ConvertImageDtype, RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm, trange

sys.path.insert(0, "..")
sys.path.insert(0, "../GaussianProxy")
from GaussianProxy.utils.data import BaseDataset, RandomRotationSquareSymmetry
from GaussianProxy.utils.misc import generate_all_augs

torch.set_grad_enabled(False)
sns.set_theme(context="paper")

install()

# Dataset to use
from my_conf.dataset.BBBC021_196_docetaxel_inference import BBBC021_196_docetaxel_inference as dataset  # noqa: E402

database_path = Path(dataset.path)
assert dataset.dataset_params is not None
print(f"Using dataset {dataset.name} from {database_path}")
subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
subdirs.sort(key=dataset.dataset_params.sorting_func)
print(f"Found {len(subdirs)} classes: {', '.join(e.name for e in subdirs)}")


def augment_images(repeat_number: int):
    # split nb_repeats times the dataset into 2 non-overlapping parts, respecting classes proportions...
    transforms = Compose([ConvertImageDtype(torch.uint8)])  # no augmentations! hard-saved after
    print(f"Using transforms:\n{transforms}")
    nb_elems_per_class: dict[str, int] = {}

    # create split and datasets once
    assert dataset.dataset_params is not None
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
    print("ds1:", ds1)
    print("ds2:", ds2)

    nb_elems_per_class["all_classes"] = sum(nb_elems_per_class.values())
    print("nb_elems_per_class:", nb_elems_per_class)

    # Augment the datasets on disk
    this_rep_base_save_path = TMP_AUG_DS_SAVE_PATH / f"repeat_{repeat_number}"
    print(f"Will save augmented datasets to {this_rep_base_save_path}")

    def augment_to_imgs(elem: Tensor) -> list[Image.Image]:
        list_augs_tensors = generate_all_augs(
            elem, [RandomRotationSquareSymmetry, RandomHorizontalFlip, RandomVerticalFlip]
        )
        assert all(aug_tensor.dtype == torch.uint8 for aug_tensor in list_augs_tensors), "Expected uint8 tensors"
        assert all(aug_tensor.min() >= 0 and aug_tensor.max() <= 255 for aug_tensor in list_augs_tensors), (
            "Expected [0, 255] range"
        )
        list_augs_imgs = [Image.fromarray(aug_tensor.numpy().transpose(1, 2, 0)) for aug_tensor in list_augs_tensors]
        return list_augs_imgs

    for split in ("split1", "split2"):
        for subdir in subdirs:
            if not (this_rep_base_save_path / split / subdir.name).exists():
                (this_rep_base_save_path / split / subdir.name).mkdir(parents=True)

    print("Writing augmented datasets to disk...")
    pbar = tqdm(total=len(ds1) + len(ds2), unit="base image", desc="Saving augmented images", position=2)

    def process_element(split, ds, elem_idx, elem):
        elem_path = Path(ds.samples[elem_idx])
        augs = augment_to_imgs(elem)
        for aug_idx, aug in enumerate(augs):
            save_path = this_rep_base_save_path / split / elem_path.parent.name / f"{elem_path.stem}_{aug_idx}.png"
            assert save_path.parent.exists(), f"Parent {save_path.parent} does not exist"
            assert not save_path.exists(), f"Refusing to overwrite {save_path}"
            aug.save(save_path)
        pbar.update()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for split, ds in (("split1", ds1), ("split2", ds2)):
            for elem_idx, elem in enumerate(iter(ds)):
                futures.append(executor.submit(process_element, split, ds, elem_idx, elem))

    for future in concurrent.futures.as_completed(futures):
        future.result()  # raises exception if any

    pbar.close()


def check_augmented_datasets(nb_repeats: int):
    for rep in range(nb_repeats):
        nb_elems_split1_per_class = {
            class_path.name: len(
                list((TMP_AUG_DS_SAVE_PATH / f"repeat_{rep}" / "split1" / class_path.name).glob("*.png"))
            )
            for class_path in subdirs
        }
        nb_elems_split2_per_class = {
            class_path.name: len(
                list((TMP_AUG_DS_SAVE_PATH / f"repeat_{rep}" / "split2" / class_path.name).glob("*.png"))
            )
            for class_path in subdirs
        }
        assert all(nb_elems_split1_per_class.values()), "Expected non-zero number of elements"
        for class_path in subdirs:
            assert nb_elems_split1_per_class[class_path.name] - nb_elems_split2_per_class[class_path.name] <= 8, (
                f"Expected same number of elements in both splits, got:{nb_elems_split1_per_class[class_path.name]} vs {nb_elems_split2_per_class[class_path.name]} for class {class_path.name}"
            )
        assert all(
            nb_elems_this_split_per_class[cl_path.name] % 8 == 0
            for cl_path in subdirs
            for nb_elems_this_split_per_class in (nb_elems_split1_per_class, nb_elems_split2_per_class)
        ), (
            f"Expected number of elements to be a multiple of 8, got:\n{nb_elems_split1_per_class}\nand:\n{nb_elems_split2_per_class}"
        )


# FID
# ## Compute train vs train FIDs
def compute_metrics(batch_size: int, metrics_save_path: Path):
    if metrics_save_path.exists():
        raise RuntimeError(f"File {metrics_save_path} already exists, not overwriting")

    eval_metrics: dict[str, dict[str, dict[str, float]]] = {}  ## accumulate exp repeats here

    for repeat in [subdir.name for subdir in TMP_AUG_DS_SAVE_PATH.iterdir()]:
        assert repeat.startswith("repeat_"), f"Unexpected directory {repeat}"
        torch.cuda.empty_cache()

        metrics_dict: dict[str, dict[str, float]] = {}
        metrics_dict["all_classes"] = torch_fidelity.calculate_metrics(
            input1=(TMP_AUG_DS_SAVE_PATH / repeat / "split1").as_posix(),
            input2=(TMP_AUG_DS_SAVE_PATH / repeat / "split2").as_posix(),
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
            metrics_dict_cl = torch_fidelity.calculate_metrics(
                input1=(TMP_AUG_DS_SAVE_PATH / repeat / "split1" / subdir.name).as_posix(),
                input2=(TMP_AUG_DS_SAVE_PATH / repeat / "split2" / subdir.name).as_posix(),
                cuda=True,
                batch_size=batch_size,
                isc=False,
                fid=True,
                prc=False,
                verbose=True,
            )
            metrics_dict[subdir.name] = metrics_dict_cl
        # save in common dict
        eval_metrics[repeat] = metrics_dict

    if not metrics_save_path.parent.exists():
        metrics_save_path.parent.mkdir(parents=True)
    with open(metrics_save_path, "w") as f:
        json.dump(eval_metrics, f)

    return eval_metrics


TMP_AUG_DS_SAVE_PATH = Path(Path(__file__).parent, "tmp_augmented_datasets", dataset.name)
batch_size = 4096
metrics_save_path = Path(Path(__file__).parent, "evaluations", dataset.name, "eval_metrics_TEST_HARD_AUGS.json")
reaugment = False
recompute = True
NB_REPEATS = 10
print(f"Will save augmented datasets to {TMP_AUG_DS_SAVE_PATH}")
print(f"Will save metrics to {metrics_save_path}")
print("reaugment:", reaugment, "recompute:", recompute)


if __name__ == "__main__":
    # augment datasets
    if reaugment:
        inpt = input("Confirm reaugment (y/[n]):")
        if inpt != "y":
            warn(f"Will not reaugment but resuse existing augmented datasets at {TMP_AUG_DS_SAVE_PATH}")
        else:
            warn(f"Will reaugment at {TMP_AUG_DS_SAVE_PATH}")
            for rep in trange(NB_REPEATS, unit="experiment repeat"):
                augment_images(rep)
    else:
        warn(f"Will not reaugment but resuse existing augmented datasets at {TMP_AUG_DS_SAVE_PATH}")

    # check augmented datasets
    check_augmented_datasets(NB_REPEATS)

    # compute metrics
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

    class_names = list(eval_metrics["repeat_0"].keys())
    fid_scores_by_class_train = {class_name: [] for class_name in class_names}

    for exp_rep in eval_metrics.values():
        for class_name in class_names:
            fid_scores_by_class_train[class_name].append(exp_rep[class_name]["frechet_inception_distance"])

    pprint(fid_scores_by_class_train)
