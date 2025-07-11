import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from diffusers.configuration_utils import FrozenDict
from numpy import ndarray
from omegaconf import OmegaConf
from PIL import Image
from termcolor import colored
from torch import Tensor
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from wandb.sdk.wandb_run import Run as WandBRun  # pyright: ignore[reportAttributeAccessIssue]

from GaussianProxy.conf.training_conf import Config
from GaussianProxy.utils.data import RandomRotationSquareSymmetry


def create_repo_structure(
    cfg: Config,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    this_run_folder: Path,
) -> tuple[Path, Path, Path]:
    """
    The repo structure is as follows:
    ```
    cfg.exp_parent_folder
    |   <experiment_name>
    |   |   <run_name>
    |   |   |   <yyyy-mm-dd>
    |   |   |   |   <hh-mm-ss>
    |   |   |   |   |   .hydra
    |   |   |   |   |   logs.log
    |   |   |   checkpoints
    |   |   |   |   step_<x>
    |   |   |   |   step_<y>
    |   |   |   saved_model
    |   |   |   .tmp_image_generation_folder
    |   .fidelity_caches
    |   .tmp_inference_save
    |   .torch_hub_cache
    ```
    The <experiment_name>/<run_name> structure mimics that of Weight&Biases.

    Any weights are specific to a run; any run belongs to a single experiment.

    A run might have multiple <yyyy-mm-dd>/<hh-mm-ss> folders (launch times),
    each one containing a `.hydra` and `logs.log` file, if it is resumed or simply overwritten.

    Also sets the torch hub dir to `cfg.exp_parent_folder/.cache/torch/hub`.

    Must be called by all processes.
    """
    if accelerator.is_main_process:
        this_run_folder.mkdir(exist_ok=True)

    # Create a folder to save the pipeline during training
    # This folder is specific to this *run*
    models_save_folder = Path(this_run_folder, "saved_model")
    if accelerator.is_main_process:
        models_save_folder.mkdir(exist_ok=True)

    # Create a folder to save trajectories
    saved_artifacts_folder = Path(this_run_folder, "saved_artifacts")
    if accelerator.is_main_process:
        saved_artifacts_folder.mkdir(exist_ok=True)

    # Decide where to save checkpoints
    if OmegaConf.is_missing(cfg.checkpointing, "chckpt_base_path"):
        logger.info(
            f"No checkpointing *base* folder specified, using {this_run_folder}/checkpoints as checkpointing folder"
        )
        chckpt_save_path = Path(this_run_folder, "checkpoints")
    else:
        chckpt_save_path = cfg.checkpointing.chckpt_base_path / cfg.project / cfg.run_name / "checkpoints"

    # remove the checkpointing flag file if it is present at run start because of some failed cleanup
    if accelerator.is_main_process and Path(chckpt_save_path, "checkpointing_flag.txt").exists():
        logger.warning("Checkpointing flag file exists at run start: removing it")
        Path(chckpt_save_path, "checkpointing_flag.txt").unlink()

    # verify that the checkpointing folder is empty if not resuming run from a checkpoint
    # this is specific to this *run*
    if accelerator.is_main_process:
        chckpt_save_path.mkdir(exist_ok=True, parents=True)
        chckpts = list(chckpt_save_path.iterdir())
        if not cfg.checkpointing.resume_from_checkpoint and len(chckpts) > 0:
            msg = (
                "\033[1;33mTHE CHECKPOINTING FOLDER IS NOT EMPTY BUT THE CURRENT RUN WILL NOT RESUME FROM A CHECKPOINT. "
                "THIS WILL RESULT IN ERASING THE JUST-SAVED CHECKPOINTS DURING ALL TRAINING "
                "UNTIL IT REACHES THE LAST CHECKPOINTING STEP ALREADY PRESENT IN THE FOLDER.\033[0m\n"
            )
            logger.warning(msg)

    # Set torch hub cache dir to reuse pre-downloaded model weights for eval
    # (important for offline runs)
    torch_hub_path = Path(cfg.exp_parent_folder, ".cache", "torch", "hub")
    torch_hub_path.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(torch_hub_path.as_posix())

    return models_save_folder, saved_artifacts_folder, chckpt_save_path


def modify_args_for_debug(
    cfg: Config,
    logger: MultiProcessAdapter,
    wandb_tracker: WandBRun,
    is_main_process: bool,
    init_count: int,
):
    """
    Modify the configuration for quick debugging purposes.
    """
    assert cfg.debug is True, "Expected debug mode to be enabled"
    logger.warning("Debug mode: changing configuration")
    # this dict hosts the changes to be made to the configuration
    changes: dict[tuple[str, ...], int] = {
        ("dynamic", "num_train_timesteps"): 100,
        ("training", "nb_time_samplings"): init_count + 50,
        ("checkpointing", "checkpoint_every_n_steps"): 25,
        (
            "evaluation",
            "every_n_opt_steps",
        ): 15,
        ("evaluation", "nb_video_timesteps"): 3,
    }
    # now actually change the configuration,
    # updating the registered wandb config accordingly
    for param_name_tuple, new_param_value in changes.items():
        # Navigate through the levels of cfg
        cfg_level = cfg
        for level in param_name_tuple[:-1]:
            cfg_level = getattr(cfg_level, level)
            param_name = param_name_tuple[-1]
            logger.warning(f"   {param_name}: {getattr(cfg_level, param_name)} -> {new_param_value}")
            setattr(cfg_level, param_name, new_param_value)
            if is_main_process:
                wandb_tracker.config.update({"/".join(param_name_tuple): new_param_value}, allow_val_change=True)
    # now change the evaluation strategies' configurations
    for strat in cfg.evaluation.strategies:
        logger.warning(f"   {strat.name}.nb_diffusion_timesteps: {strat.nb_diffusion_timesteps} -> 3")
        strat.nb_diffusion_timesteps = 3
        if hasattr(strat, "nb_samples_to_gen_per_time"):
            # TODO move this check below to the dedicated "checks" function?
            assert hasattr(strat, "batch_size"), "Expected batch_size to be present"
            # change nb_samples_to_gen_per_time, but only if it's a hard-coded value,
            # (to keep checking the 'adapt'/'half' code paths)
            # (will be changed later in the training loop anyway)
            if isinstance(strat.nb_samples_to_gen_per_time, int):  # pyright: ignore[reportAttributeAccessIssue]
                logger.warning(
                    f"   {strat.name}.nb_samples_to_gen_per_time: {strat.nb_samples_to_gen_per_time} -> {2 * strat.batch_size}"  # pyright: ignore[reportAttributeAccessIssue]
                )
                strat.nb_samples_to_gen_per_time = 2 * strat.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(strat, "selected_times"):
            # subsample selected times to 2 (with hardcoded seed to get the same subsampling between processes!)
            chosen_times = random.Random(42).sample(strat.selected_times, 2)  # pyright: ignore[reportAttributeAccessIssue]
            logger.warning(f"   {strat.name}.selected_times: {strat.selected_times} -> {chosen_times}")  # pyright: ignore[reportAttributeAccessIssue]
            strat.selected_times = chosen_times  # pyright: ignore[reportAttributeAccessIssue]
    # TODO: update registered wandb config for evaluation strategies' configurations


def get_evenly_spaced_timesteps(nb_empirical_dists: int) -> list[float]:
    """
    Give evenly spaced timesteps between 0 and 1 to empirical distributions.
    """
    dist_timesteps = torch.linspace(0, 1, nb_empirical_dists)
    return dist_timesteps.tolist()


def sample_with_replacement(
    data: Tensor,
    batch_size: int,
    logger: MultiProcessAdapter,
    proc_idx: int,
    do_log: bool = True,
) -> Tensor:
    """
    Sample `batch_size` samples with replacement from `data`.

    The samples are drawn uniformly.
    This is a no-op if `batch_size` is equal to the number of samples in `data`.
    """
    if batch_size != data.shape[0]:
        if do_log:
            logger.warning(
                f"Resampling tensor of batch size {data.shape[0]} to {batch_size} on process {proc_idx}",
                main_process_only=False,
            )
        indices_w_replacement = torch.randint(data.shape[0], (batch_size,))
        data = data[indices_w_replacement, ...]
    return data


ACCEPTED_ARTIFACTS_NAMES_FOR_LOGGING = Literal[
    "trajectories",
    "inversions",
    "starting_samples",
    "regenerations",
    "simple_generations",
    "noised_samples",
]


@torch.inference_mode()
def save_eval_artifacts_log_to_wandb(
    tensors_to_save: Tensor,
    save_folder: Path,
    global_optimization_step: int,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    eval_strat: str,
    artifact_name: ACCEPTED_ARTIFACTS_NAMES_FOR_LOGGING,
    logging_normalization: list[str],
    max_nb_to_save_and_log: int = 16,
    captions: None | list[None] | list[str] = None,
    split_name: str | None = None,
):
    """
    Save trajectories (videos) to disk and log images and videos to W&B.

    Can be called by all processes.
    """
    # 0. Checks
    assert tensors_to_save.ndim in (
        4,
        5,
    ), f"Expected 4D or 5D tensor, got {tensors_to_save.ndim}D with shape {tensors_to_save.shape}"

    # 1. Only the `max_nb_to_save_and_log` first elements are saved & logged
    sel_to_save = tensors_to_save[:max_nb_to_save_and_log].clone()
    if captions is None:
        captions = [None] * len(tensors_to_save)
    sel_captions = captions[:max_nb_to_save_and_log]

    # 2. Save some raw images / trajectories to disk (all processes)
    if split_name is not None:
        file_path = (
            save_folder
            / eval_strat
            / artifact_name
            / split_name
            / f"step_{global_optimization_step}_proc_{accelerator.process_index}.pt"
        )
        split_name_str_repr_for_logging = split_name.split("_")[0]
    else:
        file_path = (
            save_folder
            / eval_strat
            / artifact_name
            / f"step_{global_optimization_step}_proc_{accelerator.process_index}.pt"
        )
        split_name_str_repr_for_logging = ""
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if file_path.exists():  # might exist if resuming
        file_path.unlink()  # must manually remove it as it's most probably read-only
    torch.save(sel_to_save.to("cpu", torch.float16), file_path)
    logger.debug(
        f"Saved raw samples to {file_path.parent} on process {accelerator.process_index}",
        main_process_only=False,
    )

    # 3. Log to W&B (main process only)
    if sel_to_save.shape[tensors_to_save.ndim - 3] == 1:
        # transform 1-channel images into fake 3-channel RGB images for compatibility
        logger.debug("Duplicating single channel to obtain 3-channels images")
        if tensors_to_save.ndim == 5:
            # 5D tensor: (batch, time, channels, height, width)
            sel_to_save = sel_to_save.repeat(1, 1, 3, 1, 1)
        else:
            # 4D tensor: (batch, channels, height, width)
            sel_to_save = sel_to_save.repeat(1, 3, 1, 1)
        logger.debug(f"Resulting shape: {sel_to_save.shape}")
    elif sel_to_save.shape[tensors_to_save.ndim - 3] == 4:
        # transform 4-channel images into 3-channel RGB images for compatibility
        logger.debug(f"Composing RGB images from 4-channel images at dim {tensors_to_save.ndim - 3}")
        logger.debug("WARNING: the 4th channel is discarded")
        sel_to_save = sel_to_save[:, :3]  # TODO: actually compose from 4 channels
    else:
        assert sel_to_save.shape[tensors_to_save.ndim - 3] == 3, (
            f"Expected 1, 3 or 4 channels, got {sel_to_save.shape[tensors_to_save.ndim - 3]} in total shape {sel_to_save.shape}"
        )

    # normalize images / videos for logging
    normalized_elements_for_logging = normalize_elements_for_logging(sel_to_save, logging_normalization)

    # videos case
    if tensors_to_save.ndim == 5:
        assert artifact_name == "trajectories", (
            f"Expected name to be 'trajectories' for 5D tensors, got {artifact_name}"
        )
        kwargs = {
            "fps": max(int(tensors_to_save.shape[1] / 20), 1),  # ~ 20s
            "format": "mp4",
        }
        for norm_method, normed_vids in normalized_elements_for_logging.items():
            videos = wandb.Video(normed_vids, **kwargs)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
            if split_name is not None:
                artifact_path = f"{eval_strat}/Generated videos/{split_name}/{norm_method} normalized"
            else:
                artifact_path = f"{eval_strat}/Generated videos/{norm_method} normalized"
            accelerator.log({artifact_path: videos}, step=global_optimization_step)
            logger.debug(
                f"Logged {len(sel_to_save)} {eval_strat}, {norm_method} normalized {split_name_str_repr_for_logging} trajectories to W&B at step {global_optimization_step}",
            )

    # images case
    else:
        match artifact_name:
            case "inversions":
                wandb_title = "Inverted Gaussians"
            case "starting_samples":
                wandb_title = "Starting samples"
            case "regenerations":
                wandb_title = "Regenerated samples"
            case "simple_generations":
                wandb_title = "Pure sample generations"
            case "noised_samples":
                wandb_title = "Slightly noised starting samples"
            case _:
                raise ValueError(
                    f"Unknown name: {artifact_name}; expected one of 'inversions' or 'starting_samples' for 4D tensors"
                )
        for norm_method, normed_imgs in normalized_elements_for_logging.items():
            # PIL RGB mode expects 3x8-bit pixels in (H, W, C) format
            images = [Image.fromarray(image.transpose(1, 2, 0), mode="RGB") for image in normed_imgs]
            if split_name is not None:
                artifact_path = f"{eval_strat}/{wandb_title}/{split_name}/{norm_method} normalized"
            else:
                artifact_path = f"{eval_strat}/{wandb_title}/{norm_method} normalized"
            accelerator.log(
                {
                    artifact_path: [
                        wandb.Image(image, caption=sel_captions[img_idx])  # pyright: ignore[reportAttributeAccessIssue]
                        for img_idx, image in enumerate(images)
                    ]
                },
                step=global_optimization_step,
            )
        logger.info(
            f"Logged {len(sel_to_save)} {split_name_str_repr_for_logging} {wandb_title[0].lower() + wandb_title[1:]} to W&B at step {global_optimization_step}",
        )


@torch.inference_mode()
def normalize_elements_for_logging(elems: Tensor, logging_normalization: list[str]) -> dict[str, ndarray]:
    """
    Normalize [-1; 1] images or videos for logging to W&B.

    Output range and type is *always* `[0;255]` and `np.uint8`.
    """
    # 1. Determine the dimensions for normalization based on the number of dimensions in the tensor
    match elems.ndim:
        case 5:
            per_image_norm_dims = (2, 3, 4)
        case 4:
            per_image_norm_dims = (1, 2, 3)
        case _:
            raise ValueError(f"Unsupported number of dimensions. Expected 4 or 5, got shape {elems.shape}")
    # 2. Normalize the elements for logging
    normalized_elems_for_logging = {}
    for norm in logging_normalization:
        # normalize to [0;1] using some method
        match norm:
            case "image min-max":
                norm_elems = elems.clone() - elems.amin(dim=per_image_norm_dims, keepdim=True)
                norm_elems /= norm_elems.amax(dim=per_image_norm_dims, keepdim=True)
            case "image 5perc-95perc":
                norm_elems = elems.clone().cpu().numpy()  # torch doesn't allow quantile computation over multiple dims
                norm_elems -= np.percentile(norm_elems, 5, axis=per_image_norm_dims, keepdims=True)
                norm_elems /= np.percentile(norm_elems, 95, axis=per_image_norm_dims, keepdims=True)
            case "video min-max":
                assert elems.ndim == 5, f"Expected 5D tensor for video normalization, got shape {elems.shape}"
                norm_elems = elems.clone() - elems.amin(dim=(1, 2, 3, 4), keepdim=True)
                norm_elems /= norm_elems.amax(dim=(1, 2, 3, 4), keepdim=True)
            case "-1_1 raw":
                norm_elems = elems.clone() / 2
                norm_elems += 0.5
            case "-1_1 clipped":
                norm_elems = elems.clone() / 2
                norm_elems += 0.5
                norm_elems = norm_elems.clamp(0, 1)
            case _:
                raise ValueError(f"Unknown normalization: {norm}")

        # convert to [0;255] np.uint8 arrays for wandb / PIL
        if isinstance(norm_elems, Tensor):
            norm_elems = norm_elems.to(torch.float32).cpu().numpy()

        norm_elems = (norm_elems * 255).astype(np.uint8)
        normalized_elems_for_logging[norm] = norm_elems

    return normalized_elems_for_logging


def bold(s: str | int) -> str:
    return colored(s, None, None, ["bold"])


def warn_about_dtype_conv(
    model: torch.nn.Module,
    target_dtype: torch.dtype,
    logger: Logger | MultiProcessAdapter,
):
    if model.dtype == torch.bfloat16 and target_dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        logger.warning(f"model is bf16 but target_dtype is {target_dtype}: conversion might be weird/invalid")


def save_images_for_metrics_compute(
    images: Tensor,
    save_folder: Path,
    file_extension: str,
    process_idx: int | None = None,
):
    """
    Saves [-1;1] tensors to [0; 255] uint8 3 channels images in the given folder.

    TODO(if bottleneck): make this non-blocking and check all results at once
    at very end of all generations in main `_metrics_computation` func.
    """
    # Checks
    assert images.ndim == 4, f"Expected 4D tensor, got {images.shape}"
    assert images.dtype == torch.float32, f"Expected float32 tensor, got {images.dtype}"
    save_folder.mkdir(parents=True, exist_ok=True)

    # Normalize images to [0;255] np.uint8 range
    if not (images.min() >= -1 and images.max() <= 1):
        print(f"Expected [-1;1] range, got {images.min()} to {images.max()}")
    normalized_imgs = normalize_elements_for_logging(images, ["-1_1 raw"])["-1_1 raw"]

    def _save_image_wrapper(img: ndarray):
        img = img.transpose(1, 2, 0)  # channel last
        if img.shape[2] == 1:  # save images as 3 channels for metrics computation
            img = np.tile(img, (1, 1, 3))  # (H, W, 3)
        pil_img = Image.fromarray(img)
        tmpst = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if process_idx is not None:
            filename = f"proc_{process_idx}_{tmpst}.{file_extension}"
        else:
            filename = tmpst + f".{file_extension}"
        assert not (save_folder / filename).exists(), f"File {filename} already exists"
        pil_img.save(save_folder / filename)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_save_image_wrapper, img) for img in normalized_imgs]

        for future in as_completed(futures):
            future.result()  # raises


def verify_model_configs(
    loaded_config: FrozenDict | dict,
    declared_config: FrozenDict | dict,
    model_name: str,
):
    """Compare model configurations excluding internal fields.

    Args:
        loaded_config: (FrozenDict) from loaded model
        declared_config: (FrozenDict) from declaration in config
        model_name: (str) for error messages

    Raises:
        ValueError if configs don't match
    """

    def filter_internal_fields(config_dict: FrozenDict | dict):
        """Remove internal fields (starting with '_') from FrozenDict config dictionary."""
        return {k: v for k, v in dict(config_dict).items() if not k.startswith("_")}

    filtered_loaded = filter_internal_fields(loaded_config)
    filtered_declared = filter_internal_fields(declared_config)

    if filtered_loaded != filtered_declared:
        # Find differing keys
        all_keys = set(filtered_loaded.keys()) | set(filtered_declared.keys())
        diffs = []
        for key in sorted(all_keys):
            if key not in filtered_loaded:
                diffs.append(f"Missing in loaded: {key}")
            elif key not in filtered_declared:
                diffs.append(f"Missing in declared: {key}")
            elif filtered_loaded[key] != filtered_declared[key]:
                diffs.append(f"Different value for {key}:")
                diffs.append(f"  Loaded:   {filtered_loaded[key]}")
                diffs.append(f"  Declared: {filtered_declared[key]}")

        raise ValueError(f"The config of the loaded {model_name} does not match the declared one:\n" + "\n".join(diffs))


SUPPORTED_TRANSFORMS_TO_GENERATE = (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotationSquareSymmetry,
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotationSquareSymmetry",
)


@overload
def generate_all_augs(img: Tensor, transforms: list[type] | list[str]) -> list[Tensor]: ...


@overload
def generate_all_augs(img: Image.Image, transforms: list[type] | list[str]) -> list[Image.Image]: ...


def generate_all_augs(
    img: Tensor | Image.Image, transforms: list[type] | list[str]
) -> list[Tensor] | list[Image.Image]:
    """Generate all augmentations of `img` based on `transforms`."""
    # general checks
    assert all(t in SUPPORTED_TRANSFORMS_TO_GENERATE for t in transforms), f"Unsupported transforms: {transforms}"

    # choose backend
    if isinstance(img, Tensor):
        aug_imgs = _generate_all_augs_torch(img, transforms)
    elif isinstance(img, Image.Image):
        aug_imgs = _generate_all_augs_pil(img, transforms)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    assert len(aug_imgs) in (
        1,
        2,
        4,
        8,
    ), f"Expected 1, 2, 4, or 8 images at this point, got {len(aug_imgs)}"

    return aug_imgs


# TODO: factorize backends better


def _generate_all_augs_torch(img: Tensor, transforms: list[type] | list[str]) -> list[Tensor]:
    """torch backend for `generate_all_augs`."""
    assert img.ndim == 3, f"Expected 3D image, got shape {img.shape}"
    # generate all possible augmentations
    aug_imgs = [img]
    if RandomHorizontalFlip in transforms or "RandomHorizontalFlip" in transforms:
        aug_imgs.append(torch.flip(img, [2]))
    if RandomVerticalFlip in transforms or "RandomVerticalFlip" in transforms:
        for base_img_idx in range(len(aug_imgs)):
            aug_imgs.append(torch.flip(aug_imgs[base_img_idx], [1]))
    if RandomRotationSquareSymmetry in transforms or "RandomRotationSquareSymmetry" in transforms:
        if len(aug_imgs) == 4:  # must have been flipped in both directions then
            aug_imgs += [
                torch.rot90(aug_imgs[0], k=1, dims=(1, 2)),
                torch.rot90(aug_imgs[0], k=3, dims=(1, 2)),
            ]
            aug_imgs += [
                torch.rot90(aug_imgs[1], k=1, dims=(1, 2)),
                torch.rot90(aug_imgs[1], k=3, dims=(1, 2)),
            ]
        elif len(aug_imgs) in (1, 2):  # simply perform all 3 rotations on each image
            for base_img_idx in range(len(aug_imgs)):
                for nb_rot in [1, 2, 3]:
                    aug_imgs.append(torch.rot90(aug_imgs[base_img_idx], k=nb_rot, dims=(1, 2)))
        else:
            raise ValueError(f"Expected 1, 2, or 4 images at this point, got {len(aug_imgs)}")

    return aug_imgs


def _generate_all_augs_pil(img: Image.Image, transforms: list[type] | list[str]) -> list[Image.Image]:
    """PIL backend for `generate_all_augs`."""
    # generate all possible augmentations
    aug_imgs = [img]
    if RandomHorizontalFlip in transforms or "RandomHorizontalFlip" in transforms:
        aug_imgs.append(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    if RandomVerticalFlip in transforms or "RandomVerticalFlip" in transforms:
        for base_img_idx in range(len(aug_imgs)):
            aug_imgs.append(aug_imgs[base_img_idx].transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    if RandomRotationSquareSymmetry in transforms or "RandomRotationSquareSymmetry" in transforms:
        if len(aug_imgs) == 4:  # must have been flipped in both directions then
            aug_imgs += [
                aug_imgs[0].transpose(Image.Transpose.ROTATE_90),
                aug_imgs[0].transpose(Image.Transpose.ROTATE_270),
            ]
            aug_imgs += [
                aug_imgs[1].transpose(Image.Transpose.ROTATE_90),
                aug_imgs[1].transpose(Image.Transpose.ROTATE_270),
            ]
        elif len(aug_imgs) in (1, 2):  # simply perform all 3 rotations on each image
            for base_img_idx in range(len(aug_imgs)):
                aug_imgs += [
                    aug_imgs[base_img_idx].transpose(Image.Transpose.ROTATE_90),
                    aug_imgs[base_img_idx].transpose(Image.Transpose.ROTATE_180),
                    aug_imgs[base_img_idx].transpose(Image.Transpose.ROTATE_270),
                ]
        else:
            raise ValueError(f"Expected 1, 2, or 4 images at this point, got {len(aug_imgs)}")

    return aug_imgs


def hard_augment_dataset_all_square_symmetries(
    dataset_path_or_paths: Path | list[Path],
    logger: MultiProcessAdapter,
    files_ext: str,
    max_workers: int | None = None,
    augmentations: list[type] | list[str] = (
        RandomHorizontalFlip,
        RandomVerticalFlip,
        RandomRotationSquareSymmetry,
    ),  # type: ignore[reportArgumentType]
):
    """
    Save ("in-place") the n augmented versions of each image in the given `dataset_path`.

    ### Args:
    - `dataset_path_or_paths` (`Path` or `list[Path]`): The path or list of paths to the dataset(s) to augment.

    Each image will be augmented with up to the 8 square symmetries (Dih4)
    and saved in the same subfolder with '_aug_<idx>' appended.

    This function launches a pool of `max_workers` processes.
    """
    # Get all base images
    if isinstance(dataset_path_or_paths, Path):
        all_base_imgs = list(dataset_path_or_paths.rglob(f"*.{files_ext}"))
    elif isinstance(dataset_path_or_paths, list):
        all_base_imgs = []
        for dataset_path in dataset_path_or_paths:
            all_base_imgs += list(dataset_path.rglob(f"*.{files_ext}"))
    else:
        raise TypeError(f"Expected Path or list[Path], got {type(dataset_path_or_paths)}")

    assert len(all_base_imgs) > 0, f"No '*.{files_ext}' images found in {dataset_path_or_paths}"
    logger.debug(f"Found {len(all_base_imgs)} base images in {dataset_path_or_paths}")

    # Save augmented images
    logger.debug("Writing augmented datasets to disk")

    futures = []
    with ProcessPoolExecutor(max_workers) as executor:
        for base_img in all_base_imgs:
            futures.append(executor.submit(_aug_save_img, base_img, augmentations))

        for future in as_completed(futures):
            future.result()  # raises exception if any

    logger.debug("Finished augmenting datasets to disk")


def _aug_save_img(base_img_path: Path, augmentations: list[type] | list[str]):
    base_img = Image.open(base_img_path)
    augs = generate_all_augs(base_img, augmentations)
    for aug_idx, aug in enumerate(augs[1:]):  # skip the original image
        save_path = base_img_path.parent / f"{base_img_path.stem}_aug{aug_idx}{base_img_path.suffix}"
        aug.save(save_path)
