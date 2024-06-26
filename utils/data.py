from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from accelerate.logging import MultiProcessAdapter
from hydra.utils import instantiate
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from conf.conf import Config


def _biotine_2D_image_builder(
    cfg: Config,
    num_workers: int,
    logger: MultiProcessAdapter,
    debug: bool = False,
    train_split_frac: float = 0.9,
    expected_initial_data_range: tuple[float, float] = (0, 1),
) -> tuple[dict[int, DataLoader], dict[int, DataLoader]]:
    """Returns a list of dataloaders for the biotine data (images directly): one dataloader per timestamp.

    Each dataloader is a `torch.utils.data.DataLoader` over a custom `NumpyDataset`.

    Raw `npy` images are expected to be found in `cfg.path` with the following structure:
    ```
    |   - cfg.dataset.path
    |   |   - 1
    |   |   |   - image_time_1_id_1.npy
    |   |   |   - image_time_1_id_2.npy
    |   |   |   - ...
    |   |   |   - image_time_1_id_8451.npy
    |   |   - 2
    |   |   |   - image_time_2_id_1.npy
    |   |   |   - ...
    |   |   - ...
    ```
    """
    database_path = Path(cfg.dataset.path)
    subdirs = sorted([e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")])

    files_dict_per_time: dict[int, list[Path]] = {}
    for subdir in subdirs:
        subdir_files = sorted(subdir.glob("*.npy"))
        timestep = subdir.name
        assert timestep.isdigit(), "Subdir name should be a number"
        timestep = int(timestep)
        files_dict_per_time[timestep] = subdir_files

    # sort the files by timestamp
    files_dict_per_time = dict(sorted(files_dict_per_time.items()))
    logger.debug(f"Found {sum([len(f) for f in files_dict_per_time.values()])} files in total")

    # selected files
    files_dict_per_time = {k: v for k, v in files_dict_per_time.items() if k in cfg.dataset.selected_dists}
    assert (
        files_dict_per_time is not None and len(files_dict_per_time) >= 2
    ), f"No or less than 2 times selected: cfg.dataset.selected_dists is {cfg.dataset.selected_dists} resulting in selected timesteps {list(files_dict_per_time.keys())}"
    logger.info(f"Selected {len(files_dict_per_time)} timesteps")

    # Build train datasets & test dataloaders
    train_dataloaders_dict = {}
    test_dataloaders_dict = {}
    transforms = instantiate(cfg.dataset.transforms)
    logger.warning(f"Using transforms: {transforms} over expected initial data range={expected_initial_data_range}")
    if debug:
        logger.warning(">>> DEBUG MODE: LIMITING TEST DATALOADER TO 1 BATCH <<<")
    # Time per time
    for timestamp, files in files_dict_per_time.items():
        if debug:
            # test_idxes are different between processes but that's fine for debug
            test_idxes = (
                np.random.default_rng().choice(len(files), cfg.training.train_batch_size, replace=False).tolist()
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
        train_ds = NumpyDataset(train_files, transforms, expected_initial_data_range)
        assert (
            train_ds[0].shape == cfg.dataset.data_shape
        ), f"Expected data shape of {cfg.dataset.data_shape} but got {train_ds[0].shape}"
        logger.info(f"Built train dataset for timestamp {timestamp}:\n{train_ds}")
        # batch_size does *NOT* correspond to the actual train batch size
        # *Time-interpolated* batches will be manually built afterwards!
        train_dataloaders_dict[timestamp] = DataLoader(
            train_ds,
            batch_size=max(1, cfg.training.train_batch_size // 2),
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=cfg.dataloaders.train_prefetch_factor,
            pin_memory=cfg.dataloaders.pin_memory,
            persistent_workers=cfg.dataloaders.persistent_workers,
        )
        # Create test dataloader
        test_ds = NumpyDataset(test_files, transforms, expected_initial_data_range)
        assert (
            test_ds[0].shape == cfg.dataset.data_shape
        ), f"Expected data shape of {cfg.dataset.data_shape} but got {test_ds[0].shape}"
        logger.info(f"Built test dataset for timestamp {timestamp}:\n{test_ds}")
        test_dataloaders_dict[timestamp] = DataLoader(
            test_ds,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,  # keep the order for consistent logging
        )
    return train_dataloaders_dict, test_dataloaders_dict


class NumpyDataset(Dataset):
    """Just a dataset loading NumPy arrays."""

    def __init__(
        self,
        samples: list[str] | list[Path],
        transforms: Callable,
        expected_initial_data_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.transforms = transforms
        self.expected_initial_data_range = expected_initial_data_range

    def _loader(self, path: str | Path) -> Tensor:
        t = torch.from_numpy(np.load(path))
        if self.expected_initial_data_range is not None:
            if t.min() < self.expected_initial_data_range[0] or t.max() > self.expected_initial_data_range[1]:
                raise ValueError(
                    f"Expected initial data range {self.expected_initial_data_range} but got [{t.min()}, {t.max()}] at {path}"
                )
        t = self.transforms(t)
        return t

    def __getitem__(self, index: int) -> Tensor:
        path = self.samples[index]
        sample = self._loader(path)
        return sample

    def __getitems__(self, indexes: list[int]) -> list[Tensor]:
        paths = [self.samples[idx] for idx in indexes]
        samples = [self._loader(p) for p in paths]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = f"Number of datapoints: {self.__len__()}"
        lines = [head] + [" " * 4 + body]
        return "\n".join(lines)


def setup_dataloaders(
    cfg: Config, num_workers: int, logger: MultiProcessAdapter, debug: bool = False
) -> tuple[dict[int, DataLoader], dict[int, DataLoader]]:
    match cfg.dataset.name:
        case "biotine_image":
            return _biotine_2D_image_builder(cfg, num_workers, logger, debug)
        case _:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")
