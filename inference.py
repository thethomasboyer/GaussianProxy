# Imports
import logging
import sys
from math import ceil
from pathlib import Path
from typing import Optional

import colorlog
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torchvision
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from enlighten import Manager, get_manager
from numpy import ndarray
from PIL import Image
from rich.traceback import install
from torch import IntTensor, Tensor

from conf.inference_conf import InferenceConfig
from conf.training_conf import (
    ForwardNoising,
    ForwardNoisingLinearScaling,
    InvertedRegeneration,
    IterativeInvertedRegeneration,
    SimpleGeneration,
)
from my_conf.my_inference_conf import inference_conf
from utils.data import NumpyDataset
from utils.misc import _normalize_elements_for_logging
from utils.models import VideoTimeEncoding

# No grads
torch.set_grad_enabled(False)

# Nice tracebacks
install()

# logging
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

log_file_path = inference_conf.output_dir / "logs.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
file_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)
logger.addHandler(term_handler)
logger.addHandler(file_handler)

# Convert str dtype to torch symbol here to avoid loading torch at configuration time
torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[inference_conf.dtype]


def main(cfg: InferenceConfig) -> None:
    logger.info("\n" + "#" * 120 + "\n#" + " " * 47 + "Starting inference script" + " " * 46 + "#\n" + "#" * 120)
    logger.info(f"Using device {cfg.device}")
    ###############################################################################################
    #                                          Check paths
    ###############################################################################################
    project_path = cfg.root_experiments_path / cfg.project_name
    assert project_path.exists(), f"Project path {project_path} does not exist."

    run_path = project_path / cfg.run_name
    assert run_path.exists(), f"Run path {run_path} does not exist."
    logger.info(f"run path: {run_path}")

    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"output dir: {cfg.output_dir}")

    ###############################################################################################
    #                                          Load Model
    ###############################################################################################
    # denoiser
    net: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # type: ignore
        run_path / "saved_model" / "net", torch_dtype=torch_dtype
    )
    net.to(cfg.device)
    if cfg.compile:
        net = torch.compile(net)  # type: ignore

    # time encoder
    video_time_encoder: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(  # type: ignore
        run_path / "saved_model" / "video_time_encoder", torch_dtype=torch_dtype
    )
    video_time_encoder.to(cfg.device)

    # dynamic
    dynamic: DDIMScheduler = DDIMScheduler.from_pretrained(run_path / "saved_model" / "dynamic")

    ###############################################################################################
    #                                     Load Starting Images
    ###############################################################################################
    database_path = Path(cfg.dataset.path)
    subdirs = sorted([e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")])

    # we are only interested in the first subdir: timestep 0
    assert subdirs[0].name == "1", f"Expected '1' as first subdir, got {subdirs[0].name} in subdir list: {subdirs}"

    # I reuse the NumpyDataset class from the training script to load the images consistently
    starting_samples = list(subdirs[0].glob("*.npy"))
    starting_ds = NumpyDataset(starting_samples, cfg.dataset.transforms)
    logger.info(f"Built dataset from {database_path}/1:\n{starting_ds}")
    logger.info(f"Using transforms:\n{cfg.dataset.transforms}")

    # select starting samples to generate from
    if cfg.plate_name_to_simulate is not None:
        # if a plate was given, select nb_generated_samples from it, in order
        glob_pattern = f"{cfg.plate_name_to_simulate}_time_1_patch_*_*.npy"
        # assuming they are sorted in (x,y) dictionary order
        all_matching_sample_names = list(Path(cfg.dataset.path, "1").glob(glob_pattern))
        assert len(all_matching_sample_names) > 0, f"No samples found with glob pattern {glob_pattern}"
        logger.info(f"Found {len(all_matching_sample_names)} patches from plate {cfg.plate_name_to_simulate}")

        sample_names = all_matching_sample_names[: cfg.nb_generated_samples]
        assert (
            len(sample_names) == cfg.nb_generated_samples
        ), f"Expected to get at least {cfg.nb_generated_samples} samples, got {len(sample_names)}"
        logger.info(f"Selected {len(sample_names)} samples to run inference from.")
        tensors: list[Tensor] = starting_ds.get_items_by_name(sample_names)
    else:
        # if not, select nb_generated_samples samples randomly
        sample_idxes: list[int] = (
            np.random.default_rng().choice(len(starting_ds), cfg.nb_generated_samples, replace=False).tolist()
        )
        tensors = starting_ds.__getitems__(sample_idxes)
        logger.info(f"Selected {len(sample_idxes)} samples to run inference from at random.")

    starting_batch = torch.stack(tensors).to(cfg.device, torch_dtype)
    logger.debug(f"Using starting data of shape {starting_batch.shape} and type {starting_batch.dtype}")

    ###############################################################################################
    #                                       Inference passes
    ###############################################################################################
    pbar_manager: Manager = get_manager()  # type: ignore

    for eval_strat_idx, eval_strat in enumerate(cfg.evaluation_strategies):
        logger.info(f"Running evaluation strategy {eval_strat_idx+1}/{len(cfg.evaluation_strategies)}:\n{eval_strat}")
        logger.name = eval_strat.name
        if type(eval_strat) is SimpleGeneration:
            simple_gen(cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager)
        elif type(eval_strat) is ForwardNoising:
            forward_noising(cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager)
        elif type(eval_strat) is ForwardNoisingLinearScaling:
            forward_noising_linear_scaling(
                cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager
            )
        elif type(eval_strat) is InvertedRegeneration:
            inverted_regeneration(cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager)
        elif type(eval_strat) is IterativeInvertedRegeneration:
            iterative_inverted_regeneration(
                cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager
            )
        else:
            raise ValueError(f"Unknown evaluation strategy {eval_strat}")


def simple_gen(
    cfg: InferenceConfig,
    eval_strat: SimpleGeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    """
    Just simple generations.

    `batch` is ignored!
    """
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 2. Generate the time encodings
    eval_video_times = torch.rand(len(batch), device=cfg.device, dtype=torch_dtype)
    eval_video_times = torch.sort(eval_video_times).values  # sort it for better viz
    random_video_time_enc = video_time_encoder.forward(eval_video_times)

    # 3. Generate a sample
    gen_pbar = pbar_manager.counter(
        total=len(inference_scheduler.timesteps),
        position=2,
        desc="Generating samples",
        leave=False,
    )
    gen_pbar.refresh()

    image = torch.randn_like(batch)

    for t in inference_scheduler.timesteps:
        model_output: Tensor = net.forward(
            image, t, encoder_hidden_states=random_video_time_enc.unsqueeze(1), return_dict=False
        )[0]
        image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        gen_pbar.update()

    gen_pbar.close()

    save_images_or_videos(
        image,
        base_save_path,
        "simple_generations",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )


def forward_noising(
    cfg: InferenceConfig,
    eval_strat: ForwardNoising,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup scheduler
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Save the to-be noised images
    save_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )

    # 2. Sample Gaussian noise and noise the images until some step
    noise = torch.randn_like(batch)
    noise_timestep_idx = int((1 - eval_strat.forward_noising_frac) * len(inference_scheduler.timesteps))
    noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
    msg = (
        f"Adding noise until timestep {noise_timestep} (index {noise_timestep_idx}/{len(inference_scheduler.timesteps)}"
    )
    msg += f", timesteps range: ({inference_scheduler.timesteps.min().item()}, {inference_scheduler.timesteps.max().item()}))"
    logger.debug(msg)
    noise_timesteps: IntTensor = torch.full(  # type: ignore
        (batch.shape[0],),
        noise_timestep,
        device=batch.device,
        dtype=torch.int64,
    )
    slightly_noised_sample = inference_scheduler.add_noise(batch, noise, noise_timesteps)
    save_images_or_videos(
        slightly_noised_sample,
        base_save_path,
        "noised_samples",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # usefull if small inference batch size only
    video = []
    nb_vid_batches = ceil(cfg.nb_video_timesteps / cfg.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, cfg.nb_video_timesteps, device=cfg.device, dtype=torch_dtype)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * cfg.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * cfg.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        logger.debug(f"Processing video times from {start} to {start + len(video_time_batch)}")
        # at this point video_time_batch is at most cfg.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the actual batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        video_time_enc = video_time_enc.repeat_interleave(cfg.nb_generated_samples, dim=0)

        image = torch.cat([slightly_noised_sample.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # where len(video_time_batch) = cfg.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: Tensor = net.forward(
                image, t, encoder_hidden_states=video_time_enc.unsqueeze(1), return_dict=False
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_timesteps / nb_video_times_in_parallel) elements, each of shape
    # (len(video_time_batch) * batch_size, channels, hight, width)
    video = torch.cat(video)  # (nb_video_timesteps * nb_generated_samples, channels, height, width)
    video = video.split(cfg.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        cfg.nb_video_timesteps,
        cfg.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )


def forward_noising_linear_scaling(
    cfg: InferenceConfig,
    eval_strat: ForwardNoisingLinearScaling,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    # -2. Checks
    if cfg.nb_video_times_in_parallel != 1:
        logger.warning(
            f"nb_video_times_in_parallel was set to {cfg.nb_video_times_in_parallel}, but is ignored in this evaluation strategy"
        )

    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup scheduler
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Save the to-be noised images
    save_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )

    # 2. Sample Gaussian noise
    noise = torch.randn_like(batch)

    # 2.5 Misc preparations
    video = []
    video_time_pbar = pbar_manager.counter(
        total=cfg.nb_video_timesteps,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    # video times between 0 and 1
    video_times = torch.linspace(0, 1, cfg.nb_video_timesteps, device=cfg.device, dtype=torch_dtype)
    # linearly interpolate between the start and end forward_noising_fracs
    forward_noising_fracs = torch.linspace(
        eval_strat.forward_noising_frac_start,
        eval_strat.forward_noising_frac_end,
        cfg.nb_video_timesteps,
        device=cfg.device,
        dtype=torch_dtype,
    )
    logger.debug(f"Forward noising fracs: {list(forward_noising_fracs)}")

    # 3. Generate the trajectory time-per-time
    for vid_batch_idx in range(cfg.nb_video_timesteps):
        # noise until this timestep's index
        noise_timestep_idx = min(  # prevent potential OOB if forward_noising_frac_start is zero
            int((1 - forward_noising_fracs[vid_batch_idx]) * len(inference_scheduler.timesteps)),
            len(inference_scheduler.timesteps) - 1,
        )

        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        video_time = video_times[vid_batch_idx]
        video_time_enc = video_time_encoder.forward(video_time.item(), batch.shape[0])

        # noise image with forward SDE
        noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
        msg = f"Adding noise until timestep {noise_timestep} (index {noise_timestep_idx}/{len(inference_scheduler.timesteps)}"
        msg += f", timesteps range: ({inference_scheduler.timesteps.min().item()}, {inference_scheduler.timesteps.max().item()}))"
        logger.debug(msg)
        noise_timesteps: IntTensor = torch.full(  # type: ignore
            (batch.shape[0],),
            noise_timestep,
            device=batch.device,
            dtype=torch.int64,
        )
        image = inference_scheduler.add_noise(batch, noise, noise_timesteps)  # clone happens here
        # shape: (batch_size, channels, height, width)

        # denoise with backward SDE
        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: Tensor = net.forward(
                image, t, encoder_hidden_states=video_time_enc.unsqueeze(1), return_dict=False
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of nb_video_timesteps elements, each of shape (batch_size, channels, hight, width)
    video = torch.stack(video)  # (nb_video_timesteps, batch_size, channels, height, width)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        cfg.nb_video_timesteps,
        cfg.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )


def inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(dynamic.config)  # type: ignore
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Save the to-be noised images
    save_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )

    # 2. Generate the inverted Gaussians
    inverted_gauss = batch.clone()
    inversion_video_time_enc = video_time_encoder.forward(0, batch.shape[0])

    diff_time_pbar = pbar_manager.counter(
        total=len(inverted_scheduler.timesteps),
        position=1,
        desc="Diffusion timesteps" + " " * 4,
        leave=False,
    )
    diff_time_pbar.refresh()

    for t in inverted_scheduler.timesteps:
        model_output = net.forward(
            inverted_gauss, t, encoder_hidden_states=inversion_video_time_enc.unsqueeze(1), return_dict=False
        )[0]
        inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_images_or_videos(
        inverted_gauss,
        base_save_path,
        "inverted_gaussians",
        ["image min-max"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # usefull if small inference batch size only
    video = []
    nb_vid_batches = ceil(cfg.nb_video_timesteps / cfg.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, cfg.nb_video_timesteps, device=cfg.device, dtype=torch_dtype)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * cfg.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * cfg.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        logger.debug(f"Processing video times from {start} to {start + len(video_time_batch)}")
        # at this point video_time_batch is at most cfg.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the actual batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        video_time_enc = video_time_enc.repeat_interleave(cfg.nb_generated_samples, dim=0)

        image = torch.cat([inverted_gauss.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # where len(video_time_batch) = cfg.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps:
            model_output: Tensor = net.forward(
                image, t, encoder_hidden_states=video_time_enc.unsqueeze(1), return_dict=False
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_timesteps / nb_video_times_in_parallel) elements, each of shape
    # (len(video_time_batch) * batch_size, channels, hight, width)
    video = torch.cat(video)  # (nb_video_timesteps * nb_generated_samples, channels, height, width)
    video = video.split(cfg.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        cfg.nb_video_timesteps,
        cfg.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )


def iterative_inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: IterativeInvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    """
    This strategy performs iteratively:
        1. an inversion to obtain the starting Gaussian
        2. a generation from that inverted Gaussian sample to obtain the next image of the video
    over all video timesteps.

    It is thus quite costly to run...
    """
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    # thresholding is not supported by DDIMInverseScheduler; use "raw" clipping instead
    if dynamic.config["thresholding"]:
        kwargs = {"clip_sample": True, "clip_sample_range": 1}
    else:
        kwargs = {}
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(dynamic.config, **kwargs)  # pyright: ignore[ reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using inverted scheduler config: {inverted_scheduler.config}")

    # 1. Save the to-be noised images
    save_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )
    save_histogram(batch, base_save_path / "starting_samples_histogram.png", (-1, 1), 50)

    # 2. Generate the trajectory
    video = []

    video_time_pbar = pbar_manager.counter(
        total=cfg.nb_video_timesteps,
        position=1,
        desc="Video timesteps",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, cfg.nb_video_timesteps, device=cfg.device, dtype=torch_dtype)

    prev_video_time = 0
    image = batch
    for video_t_idx, video_time in enumerate(video_times):
        logger.debug(f"Video timestep index {video_t_idx + 1} / {cfg.nb_video_timesteps}")

        # 2.A Generate the inverted Gaussians
        inverted_gauss = image
        inversion_video_time = video_time_encoder.forward(prev_video_time, batch.shape[0])

        diff_time_pbar = pbar_manager.counter(
            total=len(inverted_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps (inversion)",
            leave=False,
        )
        diff_time_pbar.refresh()

        for t in inverted_scheduler.timesteps:
            model_output = net.forward(
                inverted_gauss, t, encoder_hidden_states=inversion_video_time.unsqueeze(1), return_dict=False
            )[0]
            inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        save_images_or_videos(
            inverted_gauss,
            base_save_path,
            f"inverted_gaussians_time{video_t_idx}",
            ["image min-max"],
            cfg.n_rows_displayed,
            padding=0 if cfg.plate_name_to_simulate is not None else 2,
        )
        save_histogram(inverted_gauss, base_save_path / f"inverted_gaussians_histogram_time{video_t_idx}.png", (-5, 5))

        # 2.B Generate the next image from it
        image = inverted_gauss
        video_time_enc = video_time_encoder.forward(video_time.item(), batch.shape[0])

        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps (generation)",
            leave=False,
        )
        diff_time_pbar.refresh()

        for t in inference_scheduler.timesteps:
            model_output: Tensor = net.forward(
                image, t, encoder_hidden_states=video_time_enc.unsqueeze(1), return_dict=False
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        video.append(image.clone())
        save_histogram(image, base_save_path / f"image_histogram_time{video_t_idx}.png", (-1, 1), 50)

        prev_video_time = video_time.item()
        video_time_pbar.update()

    video_time_pbar.close(clear=True)

    # video is a list of nb_video_timesteps elements, each of shape (batch_size, channels, hight, width)
    video = torch.stack(video)  # (nb_video_timesteps, batch_size, channels, height, width)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        cfg.nb_video_timesteps,
        cfg.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        cfg.n_rows_displayed,
        padding=0 if cfg.plate_name_to_simulate is not None else 2,
    )


def save_images_or_videos(
    tensor: Tensor,
    base_save_path: Path,
    artifact_name: str,
    norm_methods: list[str],
    nrows: int,
    padding: int,
):
    # Save some raw images / trajectories to disk
    file_path = base_save_path / f"{artifact_name}.pt"
    torch.save(tensor.half().cpu(), file_path)
    logger.debug(f"Saved raw {artifact_name} of shape {tensor.shape} to {file_path.name}")

    normalized_elements = _normalize_elements_for_logging(tensor, norm_methods)

    match tensor.ndim:
        case 5:  # videos
            for norm_method, normed_vids in normalized_elements.items():
                # save the videos in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.mp4"
                _save_grid_of_videos(normed_vids, save_path, nrows, padding)
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")
        case 4:  # images
            for norm_method, normed_imgs in normalized_elements.items():
                # save the images in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.png"
                _save_grid_of_images(normed_imgs, save_path, nrows, padding)
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")


def _save_grid_of_videos(videos_tensor: ndarray, save_path: Path, nrows: int, padding: int):
    # Checks
    assert videos_tensor.ndim == 5, f"Expected 5D tensor, got {videos_tensor.shape}"
    assert videos_tensor.dtype == np.uint8, f"Expected dtype uint8, got {videos_tensor.dtype}"
    assert (
        videos_tensor.min() >= 0 and videos_tensor.max() <= 255
    ), f"Expected [0;255] range, got [{videos_tensor.min()}, {videos_tensor.max()}]"

    # Convert tensor to a grid of videos
    fps = max(1, int(len(videos_tensor) / 10))
    logger.debug(f"Using fps {fps}")
    writer = imageio.get_writer(save_path, mode="I", fps=fps)

    for frame_idx, frame in enumerate(videos_tensor):
        grid_img = torchvision.utils.make_grid(torch.from_numpy(frame), nrow=nrows, padding=padding)
        pil_img = grid_img.numpy().transpose(1, 2, 0)
        logger.debug(f"Adding frame {frame_idx+1}/{len(videos_tensor)} | shape: {pil_img.shape}")
        writer.append_data(pil_img)

    writer.close()


def _save_grid_of_images(images_tensor: ndarray, save_path: Path, nrows: int, padding: int):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.dtype == np.uint8, f"Expected dtype uint8, got {images_tensor.dtype}"
    assert (
        images_tensor.min() >= 0 and images_tensor.max() <= 255
    ), f"Expected [0;255] range, got [{images_tensor.min()}, {images_tensor.max()}]"

    # Convert tensor to a grid of images
    grid_img = torchvision.utils.make_grid(torch.from_numpy(images_tensor), nrow=nrows, padding=padding)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img.numpy().transpose(1, 2, 0))
    pil_img.save(save_path)


def save_histogram(
    images_tensor: Tensor,
    save_path: Path,
    xlims: Optional[tuple[float, float]] = None,
    ymax: Optional[float] = None,
):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.shape[1] == 3, f"Expected 3 channels, got {images_tensor.shape[1]}"

    # Compute histograms per channel
    histograms = []
    for channel in range(3):
        counts, bins = np.histogram(images_tensor[:, channel].cpu().numpy().flatten(), bins=256, density=True)
        histograms.append((counts, bins))

    # Plot histograms
    plt.figure(figsize=(10, 6))
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        plt.stairs(histograms[i][0], histograms[i][1], color=color, label=f"{color} channel")
    plt.legend()
    plt.title("Histogram of Image Channels")
    suptitle = " ".join(word.capitalize() for word in save_path.stem.split("_"))
    plt.suptitle(suptitle)
    plt.xlabel("Pixel Value")
    plt.ylabel("Density")
    plt.grid()
    if xlims is not None:
        plt.xlim(xlims)
    if ymax is not None:
        plt.ylim((0, ymax))

    # Plot Gaussian distribution if "gaussian" in save_path
    xlims = plt.xlim()
    if "gaussian" in save_path.stem.lower():
        x = np.linspace(xlims[0], xlims[1], 256)
        plt.plot(x, stats.norm.pdf(x), color="grey", linestyle="dashed", label="Gaussian", alpha=0.6)

    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram to {save_path}")


if __name__ == "__main__":
    main(inference_conf)
