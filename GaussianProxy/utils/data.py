from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torchvision.transforms.functional as tf
from accelerate.logging import MultiProcessAdapter
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, dtype, float32, from_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision.transforms.v2 import Transform

from GaussianProxy.conf.training_conf import Config, DatasetParams

str_to_torch_dtype_mapping = {"fp32": float32}


class BaseDataset(Dataset):
    """Just a dataset."""

    def __init__(
        self,
        samples: list[str] | list[Path],
        transforms: Callable,
        expected_initial_data_range: Optional[tuple[float, float]] = None,
        expected_dtype: Optional[dtype] = None,  # TODO: this is never set?!?
    ) -> None:
        super().__init__()
        self.samples = samples
        self.transforms = transforms
        self.expected_initial_data_range = expected_initial_data_range
        self.expected_dtype = expected_dtype

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def load_to_pt_and_transform(self, path: str | Path) -> Tensor:
        # load data
        t = self._raw_file_loader(path)
        # checks
        if self.expected_initial_data_range is not None:
            if t.min() < self.expected_initial_data_range[0] or t.max() > self.expected_initial_data_range[1]:
                raise ValueError(
                    f"Expected initial data range {self.expected_initial_data_range} but got [{t.min()}, {t.max()}] at {path}"
                )
        if self.expected_dtype is not None:
            if t.dtype != self.expected_dtype:
                raise ValueError(f"Expected dtype {self.expected_dtype} but got {t.dtype} at {path}")
        # transform
        t = self.transforms(t)
        return t

    def __getitem__(self, index: int) -> Tensor:
        path = self.samples[index]
        sample = self.load_to_pt_and_transform(path)
        return sample

    def __getitems__(self, indexes: list[int]) -> list[Tensor]:
        paths = [self.samples[idx] for idx in indexes]
        samples = self.get_items_by_name(paths)
        return samples

    def get_items_by_name(self, names: list[str | Path] | list[str] | list[Path]) -> list[Tensor]:
        samples = [self.load_to_pt_and_transform(p) for p in names]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = self.__class__.__name__
        body_lines = [f"Number of datapoints: {self.__len__()}"]
        if self.expected_initial_data_range is not None:
            body_lines.append(f"Expected initial data range: {self.expected_initial_data_range}")
        # Indent each line in the body
        indented_body_lines = [" " * 4 + line for line in body_lines]
        return "\n".join([head] + indented_body_lines)


def _dataset_builder(
    cfg: Config,
    dataset_params: DatasetParams,
    num_workers: int,
    logger: MultiProcessAdapter,
    debug: bool = False,
    train_split_frac: float = 0.9,
) -> tuple[dict[int, DataLoader], dict[int, DataLoader]] | tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    """Builds the train & test dataloaders."""
    database_path = Path(cfg.dataset.path)
    subdirs = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    subdirs.sort(key=dataset_params.sorting_func)

    files_dict_per_time: dict[int, list[Path]] | dict[str, list[Path]] = {}
    for subdir in subdirs:
        subdir_files = sorted(subdir.glob(f"*.{dataset_params.file_extension}"))
        timestep = subdir.name
        timestep = dataset_params.key_transform(timestep)
        files_dict_per_time[timestep] = subdir_files  # pyright: ignore[reportArgumentType]
    logger.debug(f"Found {sum([len(f) for f in files_dict_per_time.values()])} files in total")

    # selected files
    if not OmegaConf.is_missing(cfg.dataset, "selected_dists") and cfg.dataset.selected_dists is not None:
        files_dict_per_time = {k: v for k, v in files_dict_per_time.items() if k in cfg.dataset.selected_dists}
        assert (
            files_dict_per_time is not None and len(files_dict_per_time) >= 2
        ), f"No or less than 2 times selected: cfg.dataset.selected_dists is {cfg.dataset.selected_dists} resulting in selected timesteps {list(files_dict_per_time.keys())}"
        logger.info(f"Selected {len(files_dict_per_time)} timesteps")
    else:
        logger.info(f"No timesteps selected, using all available {len(files_dict_per_time)}")

    # Build train datasets & test dataloaders
    train_dataloaders_dict = {}
    test_dataloaders_dict = {}
    transforms = instantiate(cfg.dataset.transforms)
    logger.warning(
        f"Using transforms: {transforms} over expected initial data range={cfg.dataset.expected_initial_data_range}"
    )
    if debug:
        logger.warning("Debug mode: limiting test dataloader to 2 evaluation batch")
    # Time per time
    for timestamp, files in files_dict_per_time.items():
        if debug:
            # test_idxes are different between processes but that's fine for debug
            test_idxes = (
                np.random.default_rng()
                .choice(
                    len(files),
                    min(2 * cfg.evaluation.batch_size, int(0.9 * len(files))),
                    replace=False,
                )
                .tolist()
            )
            test_files = [files[i] for i in test_idxes]
            train_files = [f for f in files if f not in test_files]
        else:
            # Compute the split index
            split_idx = int(train_split_frac * len(files))
            train_files = files[:split_idx]
            test_files = files[split_idx:]
        assert (
            set(train_files) | (set(test_files)) == set(files)
        ), f"Expected train_files + test_files == all files, but got {len(train_files)}, {len(test_files)}, and {len(files)} elements respectively"
        # Create train dataloader
        train_ds = dataset_params.dataset_class(train_files, transforms, cfg.dataset.expected_initial_data_range)
        assert (
            train_ds[0].shape == cfg.dataset.data_shape
        ), f"Expected data shape of {cfg.dataset.data_shape} but got {train_ds[0].shape}"
        logger.info(f"Built train dataset for timestamp {timestamp}:\n{train_ds}")
        # batch_size does *NOT* correspond to the actual train batch size
        # *Time-interpolated* batches will be manually built afterwards!
        train_dataloaders_dict[timestamp] = DataLoader(
            train_ds,
            # in average we need train_batch_size/2 samples per empirical dataset to form a train batch;
            # with this batch size, 2 empirical batches max will be needed per train batch and dataset,
            # so with at least 1 prefetch we should get decent perf
            # (at max 3 dataloader batch samplings per train batch)
            batch_size=max(1, cfg.training.train_batch_size // 2),
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=cfg.dataloaders.train_prefetch_factor,
            pin_memory=cfg.dataloaders.pin_memory,
            persistent_workers=cfg.dataloaders.persistent_workers,
        )
        # Create test dataloader
        # no flips nor rotations for consistent evaluation
        test_transforms, _ = remove_flips_and_rotations_from_transforms(transforms)
        test_ds = dataset_params.dataset_class(test_files, test_transforms, cfg.dataset.expected_initial_data_range)
        assert (
            test_ds[0].shape == cfg.dataset.data_shape
        ), f"Expected data shape of {cfg.dataset.data_shape} but got {test_ds[0].shape}"
        logger.info(f"Built test dataset for timestamp {timestamp}:\n{test_ds}")
        test_dataloaders_dict[timestamp] = DataLoader(
            test_ds,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,  # keep the order for consistent logging
        )
    return train_dataloaders_dict, test_dataloaders_dict


def remove_flips_and_rotations_from_transforms(transforms: Compose):
    """Filter out `RandomHorizontalFlip`, `RandomVerticalFlip` and `RandomRotationSquareSymmetry`."""
    is_flip_or_rotation = lambda t: isinstance(
        t, (RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry)
    )
    kept_transforms = [t for t in transforms.transforms if not is_flip_or_rotation(t)]
    removed_transforms = [type(t) for t in transforms.transforms if is_flip_or_rotation(t)]
    return Compose(kept_transforms), removed_transforms


class NumpyDataset(BaseDataset):
    """Just a dataset loading NumPy arrays."""

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        return from_numpy(np.load(path))


class ImageDataset(BaseDataset):
    """Just a dataset loading images, and moving the channel dim last."""

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        return from_numpy(np.array(Image.open(path))).permute(2, 0, 1)


def setup_dataloaders(
    cfg: Config, num_workers: int, logger: MultiProcessAdapter, debug: bool = False
) -> tuple[dict[int, DataLoader], dict[int, DataLoader]] | tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    """Returns a list of dataloaders for the dataset in cfg.dataset.name.

    Each dataloader is a `torch.utils.data.DataLoader` over a custom `Dataset`.

    Raw examples are expected to be found in `cfg.path` with the following general structure:

    ```
    |   - cfg.dataset.path
    |   |   - timestep_1
    |   |   |   - id_1.<ext>
    |   |   |   - id_2.<ext>
    |   |   |   - ...
    |   |   |   - id_6687.<ext>
    |   |   - timestep_2
    |   |   |   - id_6688.<ext>
    |   |   |   - ...
    |   |   - ...
    ```

    TODO: move DatasetParams'params into the base DataSet class used in config?
    """
    match cfg.dataset.name:
        case "biotine_image" | "biotine_image_red_channel":
            ds_params = DatasetParams(
                file_extension="npy",
                key_transform=int,
                sorting_func=lambda subdir: int(subdir.name),
                dataset_class=NumpyDataset,
            )
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
            phase_order_dict = {phase: index for index, phase in enumerate(phase_order)}
            ds_params = DatasetParams(
                file_extension="jpg",
                key_transform=str,
                sorting_func=lambda subdir: phase_order_dict[subdir.name],
                dataset_class=ImageDataset,
            )
        case "diabetic_retinopathy":
            ds_params = DatasetParams(
                file_extension="jpeg",
                key_transform=int,
                sorting_func=lambda subdir: int(subdir.name),
                dataset_class=ImageDataset,
            )
        case "ependymal_context" | "ependymal_cutout":
            ds_params = DatasetParams(
                file_extension="png",
                key_transform=int,
                sorting_func=lambda subdir: int(subdir.name),
                dataset_class=ImageDataset,
            )
        case name if name.startswith("BBBC021_"):
            ds_params = DatasetParams(
                file_extension="png",
                key_transform=str,
                sorting_func=lambda subdir: float(subdir.name),
                dataset_class=ImageDataset,
            )
        case "chromalive_tl_24h_380px":
            ds_params = DatasetParams(
                file_extension="png",
                key_transform=str,
                sorting_func=lambda subdir: int(subdir.name.split("_")[1]),
                dataset_class=ImageDataset,
            )
        case _:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    return _dataset_builder(cfg, ds_params, num_workers, logger, debug)


class RandomRotationSquareSymmetry(Transform):
    """Randomly rotate the input by a multiple of π/2."""

    def __init__(self):
        super().__init__()

    def _transform(self, inpt: Tensor, params) -> Tensor:
        rot = 90 * np.random.randint(4)
        return tf.rotate(inpt, rot)
