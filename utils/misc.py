from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from omegaconf import OmegaConf
from termcolor import colored
from torch import Tensor
from wandb.sdk.wandb_run import Run as WandBRun

from conf.conf import Config


def create_repo_structure(
    cfg: Config,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    this_run_folder: Path,
) -> tuple[Path, Path, Path]:
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

    # Create a temporary folder to save the generated trajectories during training
    # for a temporary cachefolder
    if cfg.tmp_dataloader_path is None:
        tmp_dataloader_path = Path(this_run_folder, ".tmp_dataloader")
    else:
        tmp_dataloader_path = Path(cfg.tmp_dataloader_path)

    if accelerator.is_main_process:
        tmp_dataloader_path.mkdir(exist_ok=True, parents=True)

    # Create a folder to save trajectories
    saved_trajectories_folder = Path(this_run_folder, "saved_trajectories")
    if accelerator.is_main_process:
        saved_trajectories_folder.mkdir(exist_ok=True)

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

    return (
        tmp_dataloader_path,
        models_save_folder,
        saved_trajectories_folder,
    )


def args_checker(cfg: Config, logger: MultiProcessAdapter, first_check_pass: bool = True) -> None:
    # warn if no eval_every_... arg is passed is passed
    if cfg.eval_every_n_epochs is None and cfg.eval_every_n_BI_stages is None:
        logger.warning("No evaluation will be performed during training.")
    assert (
        cfg.training.inference_batch_size >= cfg.training.train_batch_size
    ), f"Expected inference_batch_size >= train_batch_size, got {cfg.training.inference_batch_size} < {cfg.training.train_batch_size}"


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
    changes: dict[tuple[str, ...], int] = {
        ("dynamic", "num_train_timesteps"): 100,
        ("training", "nb_epochs"): 5,
        ("training", "nb_time_samplings"): 100,
        ("checkpointing", "checkpoint_every_n_steps"): 100,
        (
            "training",
            "eval_every_n_epochs",
        ): 1,
        (
            "training",
            "eval_every_n_opt_steps",
        ): 300,
    }
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


def save_trajectories_log_videos(
    tmp_dataloader_path: Path,
    saved_trajectories_folder: Path,
    nb_discretization_steps: int,
    global_optimization_step: int,
    direction: str,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    max_nb_traj: int = 10,
):
    """
    Save trajectories to disk and log videos to W&B.

    Can be called by all processes.
    """
    # Save some trajectories to disk
    traj_files = list((tmp_dataloader_path / f"proc_{accelerator.process_index}" / "trajectory").iterdir())
    assert len(traj_files) > 0, f"Did not find any trajectories to log on process {accelerator.process_index}"
    logger.debug(
        f"Found {len(list(traj_files))} trajectories to save & log on process {accelerator.process_index}",
        main_process_only=False,
    )

    sel_traj_to_save = [torch.load(p.as_posix(), map_location="cpu", mmap=True)[:max_nb_traj] for p in traj_files]
    sel_traj_to_save = torch.stack(sel_traj_to_save, dim=1)
    assert (
        nb_discretization_steps == sel_traj_to_save.shape[1]
    ), f"Expected trajectory to save to have nb_discretization_steps={nb_discretization_steps} timesteps, got {sel_traj_to_save.shape[1]}"
    this_proc_saved_trajectories_folder = saved_trajectories_folder / f"proc_{accelerator.process_index}"
    this_proc_saved_trajectories_folder.mkdir(exist_ok=True)
    torch.save(
        sel_traj_to_save,
        this_proc_saved_trajectories_folder / f"{direction}_traj_step_{global_optimization_step}.pt",
    )
    logger.info(
        f"Saved trajectories of shape {sel_traj_to_save.shape} to {this_proc_saved_trajectories_folder} on process {accelerator.process_index}",
        main_process_only=False,
    )

    # Log some videos to W&B
    assert (
        sel_traj_to_save.shape[2] == 3
    ), f"Expected trajectories to contain 3 channels for RGB images, got {sel_traj_to_save.shape[2]}"
    vids = (sel_traj_to_save.cpu().numpy() * 255).astype(np.uint8)
    kwargs = {
        "fps": max(int(nb_discretization_steps / 10), 1),  # ~ 10s
        "format": "gif",
    }
    nb_elems = len(sel_traj_to_save)
    # position
    videos = wandb.Video(vids, **kwargs)  # type: ignore
    accelerator.log(
        {f"generated_videos/{direction}": videos},
        step=global_optimization_step,
    )
    logger.info(f"Logged {nb_elems} {direction} trajectories to W&B on main process")


def _create_videos_toy_case(
    traj: Tensor,
    saved_trajectories_folder: Path,
    global_optimization_step: int,
    nb_discretization_steps: int,
) -> tuple[str, str, int]:
    assert traj.shape[3] == 2, "Expected 2 channels"
    assert traj.ndim == 4, "Expected 4D tensor"
    sel_idxes_to_log = list(range(min(1000, traj.shape[0])))
    vid_paths = []
    for channel_idx, channel_name in [(0, "positions"), (1, "velocities")]:
        sel_vids = traj[sel_idxes_to_log, :, channel_idx, ...].cpu().numpy()
        # plot the 2D points
        (saved_trajectories_folder / ".tmp").mkdir(exist_ok=True)
        for time in range(traj.shape[1]):
            fig = plt.figure()
            plt.scatter(sel_vids[:, time, 0], sel_vids[:, time, 1], s=0.15)
            plt.title(f"Time: {time}")
            if channel_name == "positions":
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)
            elif channel_name == "velocities":
                plt.xlim(-4, 4)
                plt.ylim(-4, 4)
            fig.savefig((saved_trajectories_folder / ".tmp" / f"frame_{time}.png").as_posix())
            plt.close()
        # make a video out of it
        height, width, _ = cv2.imread((saved_trajectories_folder / ".tmp" / "frame_0.png").as_posix()).shape
        (saved_trajectories_folder / f"step_{global_optimization_step}").mkdir(exist_ok=True)
        vids = (
            saved_trajectories_folder / f"step_{global_optimization_step}" / f"2d_points_video{channel_name}.webm"
        ).as_posix()
        # TODO: find a codec that doesn't output fake error messages...
        video = cv2.VideoWriter(
            vids,
            cv2.VideoWriter_fourcc(*"vp90"),  # type: ignore
            int(nb_discretization_steps / 10),
            (width, height),
        )
        for time in range(traj.shape[1]):
            video.write(cv2.imread((saved_trajectories_folder / ".tmp" / f"frame_{time}.png").as_posix()))
        video.release()
        vid_paths.append(vids)
    return *vid_paths, len(sel_idxes_to_log)


def bold(s: str | int) -> str:
    return colored(s, None, None, ["bold"])
