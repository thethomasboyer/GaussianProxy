import json
import pickle
import random
import re
from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np
import tifffile
import torch
import torchvision.transforms.functional as tf
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor, dtype
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision.transforms.v2 import Transform

from GaussianProxy.conf.training_conf import Config, DatasetParams


class BaseDataset(Dataset[Tensor]):
    """Just a dataset."""

    def __init__(
        self,
        samples: list[Path],
        transforms: Callable,
        expected_initial_data_range: Optional[tuple[float, float]] = None,
        expected_dtype: Optional[dtype] = None,  # TODO: this is never set?!?
    ) -> None:
        """
        - base_path (`Path`): the path to the dataset directory
        """
        super().__init__()
        self.samples = samples
        self.transforms = transforms
        self.expected_initial_data_range = expected_initial_data_range
        self.expected_dtype = expected_dtype
        # check that we indeed have sample
        # (# we use __str__ in the error message, so any attribute assigned below this line cannot be used by __str__)
        assert len(samples) > 0, f"Got 0 samples in {self}"
        # common base path for the dataset
        # Create train dataloader
        base_path = samples[0].parents[1]
        assert all(  # TODO: this check might take a while...
            (this_sample_base_path := f.parents[1]) == base_path for f in samples
        ), (
            f"All files should be under the same directory, got base_path={base_path} and sample_base_path={this_sample_base_path}"
        )
        self.base_path = base_path

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def load_to_pt_and_transform(self, path: str | Path) -> Tensor:
        # load data
        t = self._raw_file_loader(path)
        # checks # TODO: check if this takes too much time
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

    def __str__(self) -> str:
        head = self.__class__.__name__
        body_lines = [f"Number of datapoints: {len(self)}"]
        if self.expected_initial_data_range is not None:
            body_lines.append(f"Expected initial data range: {self.expected_initial_data_range}")
        # Indent each line in the body
        indented_body_lines = [" " * 4 + line for line in body_lines]
        return "\n".join([head] + indented_body_lines)

    def short_str(self, name: str | int) -> str:
        return f"{name}: {len(self)} samples"


class NumpyDataset(BaseDataset):
    """Just a dataset loading NumPy arrays."""

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        return torch.from_numpy(np.load(path))


class ImageDataset(BaseDataset):
    """Just a dataset loading images, and moving the channel dim last."""

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        return torch.from_numpy(np.array(Image.open(path))).permute(2, 0, 1)


class TIFFDataset(BaseDataset):
    """Just a dataset loading TIFF images, and moving the channel dim last."""

    def _raw_file_loader(self, path: str | Path) -> Tensor:
        array = tifffile.imread(path)
        if array.dtype == np.uint16:
            # torch.from_numpy does not support uint16, so convert to int32 to not loose precision
            array = array.astype(np.int32, casting="safe")
        return torch.from_numpy(array).permute(2, 0, 1)


class RandomRotationSquareSymmetry(Transform):
    """Randomly rotate the input by a multiple of π/2."""

    def __init__(self):
        super().__init__()

    def transform(self, inpt: Tensor, params) -> Tensor:
        rot = 90 * np.random.randint(4)
        return tf.rotate(inpt, rot)


TimeKey = TypeVar("TimeKey", int, str)  # TODO: remove this mess and just use str from as early as possible


def setup_dataloaders(
    cfg: Config,
    accelerator,
    num_workers: int,
    logger: MultiProcessAdapter,
    this_run_folder: Path,
    chckpt_save_path: Path,
    debug: bool = False,
) -> tuple[dict[TimeKey, DataLoader], dict[TimeKey, DataLoader]]:
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

    The train/test split that was saved to disk is reused if found
    and if `cfg.checkpointing.resume_from_checkpoint is not False`.

    TODO: move DatasetParams'params into the base DataSet class used in config
    """
    match cfg.dataset.name:
        case "biotine_image" | "biotine_image_red_channel":
            ds_params = DatasetParams(
                file_extension="npy",
                key_transform=int,
                sorting_func=lambda subdir: int(subdir.name),
                dataset_class=NumpyDataset,
            )
        case "biotine_png" | "biotine_png_hard_aug":
            ds_params = DatasetParams(
                file_extension="png",
                key_transform=int,
                sorting_func=lambda subdir: int(subdir.name),
                dataset_class=ImageDataset,
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
        case "chromaLive6h_4ch_tif_patches_380px":
            ds_params = DatasetParams(
                file_extension="tif",
                key_transform=str,
                sorting_func=lambda subdir: int(subdir.name.split("_")[1]),
                dataset_class=TIFFDataset,
            )
        case "chromaLive6h_3ch_png_patches_380px" | "chromaLive6h_3ch_png_patches_380px_hard_aug":
            ds_params = DatasetParams(
                file_extension="png",
                key_transform=str,
                sorting_func=lambda subdir: int(subdir.name.split("_")[1]),
                dataset_class=ImageDataset,
            )
        case _:
            raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    return _dataset_builder(
        cfg,
        accelerator,
        ds_params,
        num_workers,
        logger,
        this_run_folder,
        chckpt_save_path,
        debug,
    )


def _dataset_builder(
    cfg: Config,
    accelerator: Accelerator,
    dataset_params: DatasetParams,
    num_workers: int,
    logger: MultiProcessAdapter,
    this_run_folder: Path,
    chckpt_save_path: Path,
    debug: bool = False,
    train_split_frac: float = 0.9,  # TODO: set as config param and add warning if changed vs checkpoint (implies saving it)
) -> tuple[dict[TimeKey, DataLoader], dict[TimeKey, DataLoader]]:
    """Builds the train & test dataloaders."""

    # Get train & test splits for each time
    build_new_train_test_split = True
    if cfg.checkpointing.resume_from_checkpoint is not False:
        saved_splits_exist = (
            Path(this_run_folder, "train_samples.json").exists() and Path(this_run_folder, "test_samples.json").exists()
        )
        if not saved_splits_exist:
            logger.warning("No train/test split saved to disk found; building new one")
        else:
            build_new_train_test_split = False

    if build_new_train_test_split:
        all_train_files, all_test_files = _build_train_test_splits(
            cfg,
            accelerator,
            dataset_params,
            logger,
            debug,
            train_split_frac,
            chckpt_save_path,
        )
    else:
        all_train_files, all_test_files = load_train_test_splits(this_run_folder, logger)
    assert all_train_files.keys() == all_test_files.keys(), (
        f"Expected same timestamps between train and test split, got: {all_train_files.keys()} vs {all_test_files.keys()}"
    )
    timestamps = list(all_train_files.keys())

    # Build train datasets & test dataloaders
    train_dataloaders_dict = {}
    test_dataloaders_dict = {}
    if type(cfg.dataset.transforms) is not Compose:
        transforms: Compose = instantiate(cfg.dataset.transforms)
    else:
        transforms = cfg.dataset.transforms
    # remove flips and rotations if as_many_samples_as_unpaired and hard augmented dataset used
    if cfg.training.as_many_samples_as_unpaired and "_hard_augmented" not in Path(cfg.dataset.path).name:
        transforms, removed_transforms = remove_flips_and_rotations_from_transforms(transforms)
        logger.warning(
            f"as_many_samples_as_unpaired is True and '_hard_augmented' in dataset path ({cfg.dataset.path}): removed flips and rotations ({removed_transforms}) from transforms"
        )
    # no flips nor rotations for consistent evaluation
    test_transforms, _ = remove_flips_and_rotations_from_transforms(transforms)
    logger.warning(
        f"Using transforms: {transforms} over expected initial data range={cfg.dataset.expected_initial_data_range}"
    )
    logger.warning(f"Using test transforms: {test_transforms}")
    if debug:
        logger.warning("Debug mode: limiting test dataloader to 2 evaluation batch")
    # time per time
    train_reprs_to_log: list[str] = []
    test_reprs_to_log: list[str] = []
    for timestamp in timestamps:
        ### Create train dataloader
        train_files = all_train_files[timestamp]
        train_ds: BaseDataset = dataset_params.dataset_class(
            samples=train_files,
            transforms=transforms,
            expected_initial_data_range=cfg.dataset.expected_initial_data_range,
        )
        assert train_ds[0].shape == cfg.dataset.data_shape, (
            f"Expected data shape of {cfg.dataset.data_shape} but got {train_ds[0].shape}"
        )
        train_reprs_to_log.append(train_ds.short_str(timestamp))
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
        ### Create test dataloader
        test_files = all_test_files[timestamp]
        test_ds = dataset_params.dataset_class(
            samples=test_files,
            transforms=test_transforms,
            expected_initial_data_range=cfg.dataset.expected_initial_data_range,
        )
        assert test_ds[0].shape == cfg.dataset.data_shape, (
            f"Expected data shape of {cfg.dataset.data_shape} but got {test_ds[0].shape}"
        )
        test_reprs_to_log.append(test_ds.short_str(timestamp))
        test_dataloaders_dict[timestamp] = DataLoader(
            test_ds,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,  # keep the order for consistent logging
        )

    # Save train/test split to disk if new TODO: save root path of dataset only once!
    if build_new_train_test_split:
        serializable_train_files = {
            time: [p.as_posix() for p in list_paths] for time, list_paths in all_train_files.items()
        }
        with Path(this_run_folder, "train_samples.json").open("w") as f:
            json.dump(serializable_train_files, f)
        serializable_test_files = {
            time: [p.as_posix() for p in list_paths] for time, list_paths in all_test_files.items()
        }
        with Path(this_run_folder, "test_samples.json").open("w") as f:
            json.dump(serializable_test_files, f)
        logger.info("Saved new train & test samples to train_samples.json & test_samples.json")

    # print some info about the datasets
    _print_short_datasets_info(train_reprs_to_log, logger, "Train datasets:")
    _print_short_datasets_info(test_reprs_to_log, logger, "Test datasets:")

    # Return the dataloaders
    return train_dataloaders_dict, test_dataloaders_dict


def _build_train_test_splits(
    cfg: Config,
    accelerator: Accelerator,
    dataset_params: DatasetParams,
    logger: MultiProcessAdapter,
    debug: bool,
    train_split_frac: float,
    chckpt_save_path: Path,
) -> tuple[dict[TimeKey, list[Path]], dict[TimeKey, list[Path]]]:
    # Get subdirs/timestamps and sort them
    database_path = Path(cfg.dataset.path)
    subdirs = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    subdirs.sort(key=dataset_params.sorting_func)  # sort by time!

    # Get all files all times
    files_dict_per_time: dict[TimeKey, list[Path]] = {}
    for subdir in subdirs:
        subdir_files = sorted(subdir.glob(f"*.{dataset_params.file_extension}"))
        timestep = subdir.name
        timestep = dataset_params.key_transform(timestep)
        files_dict_per_time[timestep] = subdir_files  # pyright: ignore[reportArgumentType]
    tot_nb_files_found = sum([len(f) for f in files_dict_per_time.values()])
    assert tot_nb_files_found != 0, f"No files found in {database_path} with extension {dataset_params.file_extension}"
    logger.debug(f"Found {tot_nb_files_found} files in total")

    # Select times
    if not OmegaConf.is_missing(cfg.dataset, "selected_dists") and cfg.dataset.selected_dists is not None:
        files_dict_per_time = {k: v for k, v in files_dict_per_time.items() if k in cfg.dataset.selected_dists}
        assert files_dict_per_time is not None and len(files_dict_per_time) >= 2, (
            f"No or less than 2 times selected: cfg.dataset.selected_dists is {cfg.dataset.selected_dists} resulting in selected timesteps {list(files_dict_per_time.keys())}"
        )
        logger.info(f"Selected {len(files_dict_per_time)} timesteps")
    else:
        logger.info(f"No timesteps selected, using all available {len(files_dict_per_time)}")

    # Use time-unpaired data if asked for
    # (does not assume all videos span the same/total time range)
    # This runs on main process only because we want the same unpaired dataset for all processes!
    # (random sample selection happens here)
    if cfg.training.unpaired_data:
        # unpaired dataset information will be saved there
        ds_tmp_save_path = chckpt_save_path / ".unpaired_dataset_from_main.pkl"
        # Build the unpaired dataset on main
        if accelerator.is_main_process:
            logger.warning("Building time-unpaired dataset on main process")
            video_ids_times: dict[str, dict[TimeKey, Path]] = {}  # dict[video_id, dict[time, file]]
            time_key_to_time_id: dict[TimeKey, set[str]] = {time_key: set() for time_key in files_dict_per_time.keys()}
            # fill dict
            for time, files in files_dict_per_time.items():
                for f in files:
                    video_id, time_id = extract_video_id(f.stem)
                    time_key_to_time_id[time].add(time_id)
                    if video_id not in video_ids_times:
                        video_ids_times[video_id] = {time: f}
                    else:
                        assert time not in video_ids_times[video_id], (
                            f"Found multiple files at time {time} for video {video_id}: {f} and {video_ids_times[video_id][time]}"
                        )
                        video_ids_times[video_id][time] = f
            # check 1-to-1 mapping between found time ids and time keys
            assert all(len(time_key_to_time_id[time_key]) == 1 for time_key in files_dict_per_time.keys()), (
                f"Found multiple time ids for some time keys: time_key_to_time_id={time_key_to_time_id}"
            )
            # select one time at random for each video_id
            unpaired_files_dict_per_time: dict[TimeKey, list[Path]] = {}
            for video_id, times_files_d in video_ids_times.items():
                time = random.choice(list(times_files_d.keys()))
                selected_frame = times_files_d[time]
                if time not in unpaired_files_dict_per_time:
                    unpaired_files_dict_per_time[time] = [selected_frame]
                else:
                    unpaired_files_dict_per_time[time].append(selected_frame)
            # checks
            assert files_dict_per_time.keys() == unpaired_files_dict_per_time.keys(), (
                f"Some times are missing in the unpaired dataset! original: {files_dict_per_time.keys()} vs unpaired: {unpaired_files_dict_per_time.keys()}"
            )
            # re-sort per time now that we know all times are present
            unpaired_files_dict_per_time = {
                common_key: unpaired_files_dict_per_time[common_key] for common_key in files_dict_per_time.keys()
            }
            logger.info(
                f"Unpaired dataset built with {sum([len(f) for f in unpaired_files_dict_per_time.values()])} files in total"
            )
            # save this dict of Paths lists to disk
            with open(ds_tmp_save_path, "wb") as f:
                pickle.dump(unpaired_files_dict_per_time, f)
            logger.debug(f"Saved unpaired dataset to {ds_tmp_save_path} on main")
        # wait for main to finish building & saving the unpaired dataset
        accelerator.wait_for_everyone()
        with open(ds_tmp_save_path, "rb") as f:
            unpaired_files_dict_per_time = pickle.load(f)
        files_dict_per_time = unpaired_files_dict_per_time

    # Split train/test
    all_train_files: dict[TimeKey, list[Path]] = {}
    all_test_files: dict[TimeKey, list[Path]] = {}
    for timestamp, files in files_dict_per_time.items():
        if debug:
            # test_idxes are different between processes but that's fine for debug
            test_idxes = random.sample(
                range(len(files)),
                min(2 * cfg.evaluation.batch_size, int(0.9 * len(files))),
            )
            test_files = [files[i] for i in test_idxes]
            train_files = [f for f in files if f not in test_files]
        else:
            # Compute the split index
            split_idx = int(train_split_frac * len(files))
            train_files = files[:split_idx]
            test_files = files[split_idx:]
        assert set(train_files) | (set(test_files)) == set(files), (
            f"Expected train_files + test_files == all files, but got {len(train_files)}, {len(test_files)}, and {len(files)} elements respectively"
        )
        all_train_files[timestamp] = train_files
        all_test_files[timestamp] = test_files

    # use as many training samples as the hard augmented, unpaired version if asked for
    # TODO: move checks to the config composition (should always be checked anyway)
    if cfg.training.as_many_samples_as_unpaired:
        assert not cfg.training.unpaired_data, "as_many_samples_as_unpaired and unpaired_data are mutually exclusive"
        # mult_factor = 1 if using already augmented dataset, 8 otherwise
        if "_hard_augmented" in Path(cfg.dataset.path).name:
            mult_factor = 1
        else:
            mult_factor = 8  # x8 augmentation is hard-coded here
        logger.warning(
            f"Using as many samples as the unpaired, x8 hard-augmented dataset: will multiply by {round(mult_factor // len(files_dict_per_time), 2)}"
        )
        for timestep, files in files_dict_per_time.items():
            orig_nb_samples = len(files)
            nb_samples_if_unpaired = orig_nb_samples * mult_factor // len(files_dict_per_time)
            all_train_files[timestep] = random.sample(files, nb_samples_if_unpaired)

    return all_train_files, all_test_files


def load_train_test_splits(
    this_run_folder: Path, logger: MultiProcessAdapter
) -> tuple[dict[TimeKey, list[Path]], dict[TimeKey, list[Path]]]:
    # train
    with Path(this_run_folder, "train_samples.json").open("r") as f:
        all_train_files = json.load(f)
    assert isinstance(all_train_files, dict), f"Expected a dict, got {type(all_train_files)}"
    for time, list_paths in all_train_files.items():
        all_train_files[time] = [Path(p) for p in list_paths]
    # test
    with Path(this_run_folder, "test_samples.json").open("r") as f:
        all_test_files = json.load(f)
    for time, list_paths in all_test_files.items():
        all_test_files[time] = [Path(p) for p in list_paths]
    logger.info("Loaded train & test samples from train_samples.json & test_samples.json")

    return all_train_files, all_test_files


def extract_video_id(filename: str) -> tuple[str, str]:
    """
    Ugly helper to extract video_id and time from a filename, using hard-coded rules.

    TODO: include time key - finding regex in dataset config!
    """
    if m := re.search(r"_time_(\d+)_", filename):
        time: str = m.group(1)
        video_id = filename.replace(f"_time_{time}_", "_")
    elif m := re.search(r"_T(\d+)_", filename):
        time: str = m.group(1)
        video_id = filename.replace(f"_T{time}_", "_")
    else:
        raise ValueError(f"Could not extract time from filename {filename}")

    return video_id, time


def remove_flips_and_rotations_from_transforms(transforms: Compose):
    """
    Filter out `RandomHorizontalFlip`, `RandomVerticalFlip` and `RandomRotationSquareSymmetry` from `transforms`.

    ### Return
    - A new `Compose` object without the flips and rotations
    - `list` of the types of the removed transforms
    """
    is_flip_or_rotation = lambda t: isinstance(  # noqa: E731
        t, (RandomHorizontalFlip, RandomVerticalFlip, RandomRotationSquareSymmetry)
    )
    kept_transforms = [t for t in transforms.transforms if not is_flip_or_rotation(t)]
    removed_transforms = [type(t) for t in transforms.transforms if is_flip_or_rotation(t)]
    return Compose(kept_transforms), removed_transforms


# limit number of printed datasets infos because of the terminal width: TODO: make it dynamic
NB_DS_PRINTED_SINGLE_LINE = 14


def _print_short_datasets_info(reprs_to_log: list[str], logger: MultiProcessAdapter, first_message: str):
    padded_infos = []
    tot_nb_lines = 0

    for ds_idx in range(len(reprs_to_log)):
        lines = reprs_to_log[ds_idx].split("\n")
        if len(lines) > tot_nb_lines:
            tot_nb_lines = len(lines)
        max_cols = max([len(line) for line in lines])
        padded_lines = [line.ljust(max_cols) for line in lines]
        padded_infos.append("\n".join(padded_lines))

    ds_indexes_slices = [
        range(i, i + NB_DS_PRINTED_SINGLE_LINE) for i in range(0, len(reprs_to_log), NB_DS_PRINTED_SINGLE_LINE)
    ]

    complete_str = f"{first_message}\n"
    for ds_slice in ds_indexes_slices:
        for line in range(tot_nb_lines):
            this_line_all_ds = [
                padded_infos[ds_idx].split("\n")[line] for ds_idx in ds_slice if ds_idx < len(padded_infos)
            ]
            complete_str += " | ".join(this_line_all_ds) + "\n"
        complete_str += "\n"
    logger.info(complete_str)
