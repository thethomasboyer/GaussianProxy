from math import ceil
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
) -> tuple[dict[int, Dataset], dict[int, DataLoader]]:
    """Returns a list of datasets for the biotine data (images directly).

    Each dataset is a `Dataset` over a `TimeTensorsDataLoader` wrapping multiple tensors,
    and corresponding together to the empirical data at a single timepoint.

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
    train_datasets_dict = {}
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
        # Create train dataset
        train_ds = NumpyDataset(train_files, transforms, expected_initial_data_range)
        assert (
            train_ds[0].shape == cfg.dataset.data_shape
        ), f"Expected data shape of {cfg.dataset.data_shape} but got {train_ds[0].shape}"
        logger.info(f"Built train dataset for timestamp {timestamp}:\n{train_ds}")
        train_datasets_dict[timestamp] = train_ds
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
            num_workers=num_workers,
            prefetch_factor=cfg.dataloaders.prefetch_factor,
            pin_memory=cfg.dataloaders.pin_memory,
        )
    return train_datasets_dict, test_dataloaders_dict


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
) -> tuple[dict[int, Dataset], dict[int, DataLoader]]:
    match cfg.dataset.name:
        case "biotine_image":
            return _biotine_2D_image_builder(cfg, num_workers, logger, debug)
        case _:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")


class TimeTensorsDataLoader:
    """
    A data loader over a sequence of tensors.

    Not distributed (distributed logic must be handled manually).

    Can be think of as a batch loader over the concatenation of all tensors,
    but actually avoids to load them all in memory at once.
    """

    def __init__(
        self,
        tensors_paths: list[Path],
        corresponding_timesteps: list[int],
        batch_size: int,
        data_shape: torch.Size | tuple[int, ...],  # TODO: remove this need
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        logger: MultiProcessAdapter | None = None,
    ):
        if logger is not None:
            logger.debug("Instantiating TimeTensorsDataLoader")
        assert len(tensors_paths) == len(
            corresponding_timesteps
        ), f"Expected same number of tensors and times, got {len(tensors_paths)} != {len(corresponding_timesteps)}"
        self.tensors_paths = tensors_paths
        self.corresponding_timesteps = corresponding_timesteps
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        # compute the number of batches w.r.t. the true total size of the dataset,
        # lazy loading with mmap
        self.total_nb_samples = 0
        self.tensors_lengths = []
        for t_path in self.tensors_paths:
            t_len = torch.load(t_path.as_posix(), mmap=True, map_location="cpu").size(0)
            self.tensors_lengths.append(t_len)
            self.total_nb_samples += t_len
        assert sum(self.tensors_lengths) == self.total_nb_samples
        assert len(self.tensors_lengths) == len(self.tensors_paths)
        self.num_batches = ceil(self.total_nb_samples / batch_size)
        self.data_shape = data_shape
        self._cumu_nb_samples_yielded: int = 0
        self.logger = logger
        if logger is not None:
            logger.debug("Finished instantiating TimeTensorsDataLoader")

    def _run_through_tensors_build_this_batch(self, batch_idx: int) -> tuple[Tensor, Tensor]:
        """
        Slice through the tensors corresponding to this batch,
        and return the newly formed concatenated batch
        """
        start_global_idx = batch_idx * self.batch_size  # included
        end_global_idx = (batch_idx + 1) * self.batch_size  # excluded, can overflow
        batch = torch.empty((0, *self.data_shape), dtype=self.dtype, device=self.device)
        timesteps = []
        # how many total samples before the current tensor start (excluded)
        tot_samples_until_tensor_start = 0
        for t_idx, t_path in enumerate(self.tensors_paths):
            # 0. "Load" this tensor
            this_tensor_len = self.tensors_lengths[t_idx]
            # 1. Check if we have already yielded all the data
            if tot_samples_until_tensor_start >= end_global_idx:
                # won't yield any data from this tensor,
                # and actually from any further one
                break
            # 2. Check if we will yield some data from this tensor
            if tot_samples_until_tensor_start + this_tensor_len <= start_global_idx:
                # won't yield any data from this tensor
                tot_samples_until_tensor_start += this_tensor_len
                continue
            # 3. Some data from this tensor must be yielded in this batch
            # start_idx_t: where to start collecting on this tensor
            start_idx_t = max(start_global_idx - tot_samples_until_tensor_start, 0)
            # end_idx_t can overflow w/o any problem on this tensor size,
            # but we must respect the batch size!
            end_idx_t = min(
                end_global_idx - tot_samples_until_tensor_start,
                start_idx_t + self.batch_size - len(batch),
            )
            t = torch.load(t_path.as_posix(), mmap=True)
            t = t[start_idx_t:end_idx_t].to(self.device).to(self.dtype)
            tot_samples_until_tensor_start += this_tensor_len
            batch = torch.cat([batch, t])
            timesteps += [self.corresponding_timesteps[t_idx]] * len(t)
        return batch, torch.tensor(timesteps).long()

    def __iter__(self):
        for i in range(self.num_batches):
            # collect the individual data points,
            # possibly (and most probably) scattered across multiple successive tensors
            batch, timesteps = self._run_through_tensors_build_this_batch(i)
            assert (
                len(batch) == self.batch_size or i == self.num_batches - 1
            ), f"Batch {i} has size {len(batch)} != self.batch_size={self.batch_size}, yet it is not the last batch (self.num_batches={self.num_batches})"
            assert len(batch) == len(timesteps)
            yield batch, timesteps

    def __len__(self):
        return self.num_batches

    def __repr__(self):
        shortened_paths = [path.parent.name + "/" + path.name for path in self.tensors_paths]
        return (
            f"TimeTensorsDataLoader(\n"
            f"  - tensors_paths={shortened_paths}\n"
            f"  - tensors_lengths={self.tensors_lengths}\n"
            f"  - corresponding_timesteps={self.corresponding_timesteps}\n"
            f"  - batch_size={self.batch_size}\n"
            f"  - data_shape={self.data_shape}\n"
            f"  - device={self.device}\n"
            f"  - dtype={self.dtype}\n"
            f"  - num_batches={self.num_batches}\n"
            f"  - total_nb_samples={self.total_nb_samples}\n"
            f")"
        )
