"""
similarity_within_training_data.py

This script computes the similarity between the training data samples, to be used as a baseline
versus the similarity between generated and true training (/validation/test) samples.
"""

# Imports
import logging
import sys
from collections.abc import Callable
from operator import gt, lt
from pathlib import Path

import colorlog
import matplotlib.pyplot as plt
import torch
from enlighten import Manager
from torch.nn import CosineSimilarity, PairwiseDistance
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip  # noqa: F401

from GaussianProxy.utils.data import (
    BaseDataset,
    RandomRotationSquareSymmetry,  # noqa: F401
    remove_flips_and_rotations_from_transforms,
)
from GaussianProxy.utils.misc import generate_all_augs
from my_conf.dataset.BBBC021_196_docetaxel_inference import BBBC021_196_docetaxel_inference  # noqa: F401
from my_conf.dataset.biotine_image_inference import biotine_image_inference  # noqa: F401
from my_conf.dataset.diabetic_retinopathy_inference import diabetic_retinopathy_inference  # noqa: F401
from my_conf.dataset.Jurkat_inference import Jurkat_inference  # noqa: F401

torch.set_grad_enabled(False)

# Script Arguments
METRICS = ("cosine", "L2")
TRANSFORMS = [RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry]
DATASET = BBBC021_196_docetaxel_inference
DEVICE = "cuda:1"
BATCH_SIZE = 8192
BASE_SAVE_PATH = Path(".") / "datasets_similarities" / DATASET.name

# Misc. Setup
term_handler = logging.StreamHandler(sys.stdout)
term_handler.setFormatter(
    colorlog.ColoredFormatter(
        "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
)
term_handler.setLevel(logging.INFO)

log_file_path = BASE_SAVE_PATH / "logs.log"
log_file_path.parent.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
file_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)
logger.addHandler(term_handler)
logger.addHandler(file_handler)


pbar_manager = Manager()
logger.info(f"Will save outputs to {BASE_SAVE_PATH}")

# Load the dataset
assert DATASET.dataset_params is not None
database_path = Path(DATASET.path)
subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
subdirs.sort(key=DATASET.dataset_params.sorting_func)
all_samples = list(subdirs[0].parent.rglob(f"*/*.{DATASET.dataset_params.file_extension}"))
kept_transforms, removed_transforms = remove_flips_and_rotations_from_transforms(DATASET.transforms)
all_times_ds: BaseDataset = DATASET.dataset_params.dataset_class(
    all_samples,
    kept_transforms,  # no augs, they will be manually performed
    DATASET.expected_initial_data_range,
)
tot_nb_samples = len(all_times_ds)
logger.debug(f"Built all-times dataset from {subdirs[0].parent}:\n{all_times_ds}")

# Instantiate Similarities Tensors
metrics: dict[str, Callable] = dict.fromkeys(METRICS)  # pyright: ignore[reportAssignmentType]
for metric in METRICS:
    if metric == "cosine":
        metrics[metric] = lambda x, y: CosineSimilarity(dim=1, eps=1e-9)(x, y)
    elif metric == "L2":
        metrics[metric] = PairwiseDistance(p=2, eps=1e-9)
    else:
        raise ValueError(f"Unsupported metric {metric}; expected 'cosine' or 'L2'")

# get augmentation factor
aug_factor = len(generate_all_augs(all_times_ds[0], TRANSFORMS))
logger.debug(f"Augmentation factor: {aug_factor}")
all_sims = {
    metric_name: torch.full(
        (tot_nb_samples, tot_nb_samples, aug_factor),
        float("NaN"),
        device=DEVICE,
        dtype=torch.float32,
    )
    for metric_name in metrics
}

BEST_VAL = {"cosine": 0, "L2": float("inf")}
COMPARISON_OPERATORS = {
    "cosine": gt,
    "L2": lt,
}
MAX_OR_MIN = {
    "cosine": torch.maximum,
    "L2": torch.minimum,
}
worst_values = {
    metric_name: torch.full(
        (tot_nb_samples,),
        BEST_VAL[metric_name],
        device=DEVICE,
        dtype=torch.float32,
    )
    for metric_name in metrics
}
# closest_ds_idx_aug_idx[metric_name][i] = (closest_ds_idx, closest_aug_idx)
closest_ds_idx_aug_idx = {
    metric_name: torch.full(
        (tot_nb_samples, 2),
        -1,
        device=DEVICE,
        dtype=torch.int64,
    )
    for metric_name in metrics
}

num_full_batches, remaining = divmod(tot_nb_samples, BATCH_SIZE)
actual_bses = [BATCH_SIZE] * num_full_batches + ([remaining] if remaining != 0 else [])

outer_pbar = pbar_manager.counter(
    total=tot_nb_samples,
    position=1,
    desc="Training samples outer loop",
)
outer_pbar.refresh()

for batch_idx, bs in enumerate(actual_bses):
    start = batch_idx * BATCH_SIZE
    end = start + bs

    inner_pbar = pbar_manager.counter(
        total=tot_nb_samples,
        position=2,
        desc="Training samples inner loop",
        leave=False,
    )
    inner_pbar.refresh()

    # compute cosine similarities over full (augmented) all-times dataset and report largest
    base_ds_images = torch.stack(all_times_ds.__getitems__(list(range(start, end)))).to(DEVICE)

    all_times_dl = DataLoader(
        all_times_ds,
        batch_size=1,  # batch size *must* be 1 here
        num_workers=2,
        pin_memory=True,
        prefetch_factor=3,
        pin_memory_device=DEVICE,
    )

    for img_idx, img in enumerate(inner_pbar(iter(all_times_dl))):
        assert len(img) == 1, f"Expected batch size 1, got {len(img)}"
        img = img[0].to(DEVICE)
        augmented_imgs = generate_all_augs(img, TRANSFORMS)  # also take into account the augmentations!

        for aug_img_idx, aug_img in enumerate(augmented_imgs):
            tiled_aug_img = aug_img.unsqueeze(0).tile(bs, 1, 1, 1)
            for metric_name, metric in metrics.items():
                value = metric(tiled_aug_img.flatten(1), base_ds_images.flatten(1))  # pylint: disable=not-callable
                # record all similarities
                all_sims[metric_name][start:end, img_idx, aug_img_idx] = value
                # update worst found indexes
                condition = COMPARISON_OPERATORS[metric_name](value, worst_values[metric_name][start:end])
                new_idxes = torch.where(
                    condition,
                    img_idx,
                    closest_ds_idx_aug_idx[metric_name][start:end, 0],
                )
                new_aug_idxes = torch.where(
                    condition,
                    aug_img_idx,
                    closest_ds_idx_aug_idx[metric_name][start:end, 1],
                )
                closest_ds_idx_aug_idx[metric_name][start:end, 0] = new_idxes
                closest_ds_idx_aug_idx[metric_name][start:end, 1] = new_aug_idxes
                # update worst found similarities
                new_worst_values = MAX_OR_MIN[metric_name](worst_values[metric_name][start:end], value)
                worst_values[metric_name][start:end] = new_worst_values

    inner_pbar.close()
    outer_pbar.update(bs)

outer_pbar.close()
# report the largest similarities
for metric_name in metrics:
    logger.info(
        f"Worst found {metric_name} similarities: {[round(val, 3) for val in worst_values[metric_name].tolist()]}"
    )
    closest_true_imgs_names = [
        Path(all_times_ds.samples[idx]).name for idx in closest_ds_idx_aug_idx[metric_name][:, 0]
    ]
    logger.debug(f"Closest found images: {closest_true_imgs_names}")

# save all metrics and plot their histogram
for metric_name in metrics:
    this_metric_all_sims = all_sims[metric_name].cpu()
    torch.save(this_metric_all_sims, BASE_SAVE_PATH / f"all_{metric_name}.pt")
    plt.figure(figsize=(10, 6))
    plt.hist(this_metric_all_sims.flatten().numpy(), bins=300)
    plt.title(
        f"nb_samples_generated × nb_train_samples × augment_factor = {tot_nb_samples} × {tot_nb_samples} × {aug_factor} = {this_metric_all_sims.numel():,}"
    )
    plt.suptitle(f"Distribution of all {metric_name} similarities")
    plt.grid()
    plt.tight_layout()
    plt.savefig(BASE_SAVE_PATH / f"all_{metric_name}_hist.png")
