from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from numpy import ndarray
from omegaconf import OmegaConf
from PIL import Image
from termcolor import colored
from torch import Generator, Tensor
from wandb.sdk.wandb_run import Run as WandBRun

from conf.conf import Config


def create_repo_structure(
    cfg: Config,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    this_run_folder: Path,
) -> tuple[Path, Path]:
    """
    The repo structure is as follows:
    ```
    exp_parent_folder
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
    |   .fidelity_cache
    |   .torch_hub_cache
    ```
    The <experiment_name>/<run_name> structure mimics that of Weight&Biases.

    Any weights are specific to a run; any run belongs to a single experiment.

    A run might have multiple <yyyy-mm-dd>/<hh-mm-ss> folders (launch times),
    each one containing a `.hydra` and `logs.log` file, if it is resumed or simply overwritten.
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

    # verify that the checkpointing folder is empty if not resuming run from a checkpoint
    # this is specific to this *run*
    if OmegaConf.is_missing(cfg.checkpointing, "chckpt_save_path"):
        logger.info(f"No checkpointing folder specified, using {this_run_folder}/checkpoints")
        cfg.checkpointing.chckpt_save_path = Path(this_run_folder, "checkpoints")

    if accelerator.is_main_process:
        chckpt_save_path = Path(cfg.checkpointing.chckpt_save_path)
        chckpt_save_path.mkdir(exist_ok=True, parents=True)
        chckpts = list(chckpt_save_path.iterdir())
        if not cfg.checkpointing.resume_from_checkpoint and len(chckpts) > 0:
            msg = (
                "\033[1;33mTHE CHECKPOINTING FOLDER IS NOT EMPTY BUT THE CURRENT RUN WILL NOT RESUME FROM A CHECKPOINT. "
                "THIS WILL RESULT IN ERASING THE JUST-SAVED CHECKPOINTS DURING ALL TRAINING "
                "UNTIL IT REACHES THE LAST CHECKPOINTING STEP ALREADY PRESENT IN THE FOLDER.\033[0m\n"
            )
            logger.warning(msg)

    return models_save_folder, saved_artifacts_folder


def args_checker(cfg: Config, logger: MultiProcessAdapter, first_check_pass: bool = True) -> None:
    # warn if no eval_every_... arg is passed is passed
    if cfg.evaluation.every_n_epochs is None:
        logger.warning("No evaluation will be performed during training.")


def modify_args_for_debug(
    cfg: Config,
    logger: MultiProcessAdapter,
    wandb_tracker: WandBRun,
    is_main_process: bool,
):
    """
    Modify the configuration for quick debugging purposes.
    """
    assert cfg.debug is True, "Expected debug mode to be enabled"
    logger.warning(">>> DEBUG MODE: CHANGING CONFIGURATION <<<")
    # this dict hosts the changes to be made to the configuration
    changes: dict[tuple[str, ...], int] = {
        ("dynamic", "num_train_timesteps"): 100,
        ("training", "nb_epochs"): 5,
        ("training", "nb_time_samplings"): 30,
        ("checkpointing", "checkpoint_every_n_steps"): 100,
        (
            "evaluation",
            "every_n_epochs",
        ): 1,
        (
            "evaluation",
            "every_n_opt_steps",
        ): 300,
    }
    # now actually change the configuration,
    # updating the registered wandb config accordingly
    for param_name_tuple, new_param_value in changes.items():
        # Navigate through the levels of cfg
        cfg_level = cfg
        for level in param_name_tuple[:-1]:
            cfg_level = getattr(cfg_level, level)
            param_name = param_name_tuple[-1]
            logger.warning(f"{param_name}: {getattr(cfg_level, param_name)} -> {new_param_value}")
            setattr(cfg_level, param_name, new_param_value)
            if is_main_process:
                wandb_tracker.config.update({"/".join(param_name_tuple): new_param_value}, allow_val_change=True)


def camel_to_snake(name: str) -> str:
    return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip("_")


def get_evenly_spaced_timesteps(nb_empirical_dists: int) -> list[float]:
    """
    Give evenly spaced timesteps between 0 and 1 to empirical distributions.
    """
    dist_timesteps = torch.linspace(0, 1, nb_empirical_dists)
    return dist_timesteps.tolist()


def sample_with_replacement(
    data: Tensor, batch_size: int, logger: MultiProcessAdapter, proc_idx: int, do_log: bool = True
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


ACCEPTED_ARTIFACTS_NAMES_FOR_LOGGING = (
    "trajectories",
    "inversions",
    "starting_samples",
    "regenerations",
    "simple_generations",
    "noised_samples",
)


@torch.inference_mode()
def save_eval_artifacts_log_to_wandb(
    tensors_to_save: Tensor,
    save_folder: Path,
    global_optimization_step: int,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    eval_strat: str,
    artifact_name: str,
    logging_normalization: list[str],
    rng: Generator,
    max_nb_to_save_and_log: int = 16,
    captions: None | list[None] | list[str] = None,
):
    """
    Save trajectories (videos) to disk and log images and videos to W&B.

    Can be called by all processes.
    """
    # checks
    assert (
        artifact_name in ACCEPTED_ARTIFACTS_NAMES_FOR_LOGGING
    ), f"Expected name in {ACCEPTED_ARTIFACTS_NAMES_FOR_LOGGING}, got {artifact_name}"
    assert tensors_to_save.ndim in (
        4,
        5,
    ), f"Expected 4D or 5D tensor, got {tensors_to_save.ndim}D with shape {tensors_to_save.shape}"

    # Save some raw images / trajectories to disk (all processes)
    sel_idxes = torch.randint(0, len(tensors_to_save), (max_nb_to_save_and_log,), generator=rng, device=rng.device)
    sel_to_save = torch.index_select(tensors_to_save, 0, sel_idxes)
    if captions is None:
        captions = [None] * len(sel_to_save)
    file_path = save_folder / artifact_name / f"step_{global_optimization_step}_proc_{accelerator.process_index}.pt"
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if file_path.exists():  # must manually remove it as it's most probably read-only
        file_path.unlink()
    torch.save(sel_to_save, file_path)
    logger.info(
        f"Saved raw samples to {file_path.parent} on process {accelerator.process_index}", main_process_only=False
    )
    logger.debug(
        f"On process {accelerator.process_index}, data to save is: shape={sel_to_save.shape} | min={sel_to_save.min()} | max={sel_to_save.max()}",
        main_process_only=False,
    )

    # Log to W&B (main process only)
    assert (
        sel_to_save.shape[tensors_to_save.ndim - 3] == 3
    ), f"Expected trajectories to contain 3 channels at dim {tensors_to_save.ndim - 3} for RGB images, got shape {sel_to_save.shape}"
    normalized_elements_for_logging = _normalize_elements_for_logging(sel_to_save, logging_normalization)
    # videos case
    if tensors_to_save.ndim == 5:
        assert (
            artifact_name == "trajectories"
        ), f"Expected name to be 'trajectories' for 5D tensors, got {artifact_name}"
        kwargs = {
            "fps": max(int(tensors_to_save.shape[1] / 20), 1),  # ~ 20s
            "format": "mp4",
        }
        for norm_method, normed_vids in normalized_elements_for_logging.items():
            videos = wandb.Video(normed_vids, **kwargs)  # type: ignore
            accelerator.log(
                {f"{eval_strat}/Generated videos/{norm_method} normalized": videos},
                step=global_optimization_step,
            )
        logger.info(
            f"Logged {len(sel_to_save)} {eval_strat}, {norm_method} normalized trajectories to W&B on main process"
        )
    # images case (inverted Gaussians)
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
        for norm_method, normed_vids in normalized_elements_for_logging.items():
            # PIL RGB mode expects 3x8-bit pixels in (H, W, C) format
            images = [
                Image.fromarray(image.transpose(1, 2, 0), mode="RGB")
                for image in normalized_elements_for_logging[norm_method]
            ]
            accelerator.log(
                {
                    f"{eval_strat}/{wandb_title}/{norm_method} normalized": [
                        wandb.Image(image, caption=captions[img_idx]) for img_idx, image in enumerate(images)
                    ]
                },
                step=global_optimization_step,
            )
        logger.info(f"Logged {len(sel_to_save)} {wandb_title[0].lower() + wandb_title[1:]} to W&B on main process")


@torch.inference_mode()
def _normalize_elements_for_logging(elems: Tensor, logging_normalization: list[str]) -> dict[str, ndarray]:
    """
    Normalize images or videos for logging to W&B.

    Output range and type is always `[0;255]` and `np.uint8`.
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
                norm_elems = elems.clone().cpu().numpy()  # torch does allow quantile computation over multiple dims
                norm_elems -= np.percentile(norm_elems, 5, axis=per_image_norm_dims, keepdims=True)
                norm_elems /= np.percentile(norm_elems, 95, axis=per_image_norm_dims, keepdims=True)
            case "video min-max":
                assert elems.ndim == 5, f"Expected 5D tensor for video normalization, got shape {elems.shape}"
                norm_elems = elems.clone() - elems.amin(dim=(1, 2, 3, 4), keepdim=True)
                norm_elems /= norm_elems.amax(dim=(1, 2, 3, 4), keepdim=True)
            case "[-1;1] raw":
                norm_elems = elems.clone() / 2
                norm_elems += 0.5
            case "[-1;1] clipped":
                norm_elems = elems.clone() / 2
                norm_elems += 0.5
                norm_elems = norm_elems.clamp(0, 1)
            case _:
                raise ValueError(f"Unknown normalization: {norm}")
        # convert to [0;255] np.uint8 arrays for wandb / PIL
        if isinstance(norm_elems, Tensor):
            norm_elems = norm_elems.cpu().numpy()
        norm_elems = (norm_elems * 255).astype(np.uint8)
        normalized_elems_for_logging[norm] = norm_elems

    return normalized_elems_for_logging


def bold(s: str | int) -> str:
    return colored(s, None, None, ["bold"])
