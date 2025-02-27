import ast
import logging
import operator
import os
import pickle
import random
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Optional

import colorlog
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch_fidelity
from accelerate import Accelerator, DistributedType
from accelerate.logging import MultiProcessAdapter
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from enlighten import Manager, get_manager
from numpy import ndarray
from PIL import Image
from rich.traceback import install
from torch import Tensor
from torch.nn import CosineSimilarity, PairwiseDistance
from torch.profiler import (
    ProfilerActivity,
    profile,
)  # , tensorboard_trace_handler #TODO:gh-pytorch#136040
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
)
from torchvision.utils import make_grid

from GaussianProxy.conf.inference_conf import InferenceConfig
from GaussianProxy.conf.training_conf import (
    ForwardNoising,
    ForwardNoisingLinearScaling,
    InversionRegenerationOnly,
    InvertedRegeneration,
    IterativeInvertedRegeneration,
    MetricsComputation,
    SimilarityWithTrainData,
    SimpleGeneration,
)
from GaussianProxy.utils.data import (
    BaseDataset,
    remove_flips_and_rotations_from_transforms,
)
from GaussianProxy.utils.misc import (
    generate_all_augs,
    get_evenly_spaced_timesteps,
    hard_augment_dataset_all_square_symmetries,
    normalize_elements_for_logging,
    save_images_for_metrics_compute,
    warn_about_dtype_conv,
)
from GaussianProxy.utils.models import VideoTimeEncoding

# Load the config
from my_conf.my_inference_conf import inference_conf

# No grads
torch.set_grad_enabled(False)

# Speed up
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Nice tracebacks
install()


def get_distrib_logger(inference_conf: InferenceConfig) -> MultiProcessAdapter:
    """Handles distributed logging"""
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
    return MultiProcessAdapter(logger, {})


def main(cfg: InferenceConfig, logger: MultiProcessAdapter) -> None:
    ###############################################################################################
    #                                          Accelerator
    ###############################################################################################
    accelerator = Accelerator()

    ###############################################################################################
    #                                            Logger
    ###############################################################################################
    logger.info("#\n" + "#" * 120 + "\n#" + " " * 47 + "Starting inference script" + " " * 46 + "#\n" + "#" * 120)

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
    if accelerator.distributed_type == DistributedType.NO:
        device = cfg.device
        logger.info(f"Using device {device}")
    else:
        device = accelerator.device

    # denoiser
    net: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # pyright: ignore[reportAssignmentType]
        run_path / "saved_model" / "net"
    )
    nb_params_M = round(net.num_parameters() / 1e6)
    logger.info(f"Loaded denoiser from saved_model/net with ~{nb_params_M}M parameters")
    warn_about_dtype_conv(net, cfg.dtype, logger)
    net.to(device, cfg.dtype)  # pyright: ignore[reportArgumentType]
    if cfg.compile:
        net = torch.compile(net)  # pyright: ignore[reportAssignmentType]

    # time encoder
    video_time_encoder: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(  # pyright: ignore[reportAssignmentType]
        run_path / "saved_model" / "video_time_encoder"
    )
    nb_params_K = round(video_time_encoder.num_parameters() / 1e3)
    logger.info(f"Loaded video time encoder from saved_model/video_time_encoder with ~{nb_params_K}K parameters")
    warn_about_dtype_conv(video_time_encoder, cfg.dtype, logger)
    video_time_encoder.to(device, cfg.dtype)  # pyright: ignore[reportArgumentType]

    # dynamic
    orig_dynamic: DDIMScheduler = DDIMScheduler.from_pretrained(run_path / "saved_model" / "dynamic")

    if cfg.scheduler_config is not None:
        logger.info(f"Loading scheduler from {cfg.scheduler_config}")
        dynamic: DDIMScheduler = DDIMScheduler.from_pretrained(cfg.scheduler_config)
        # show the diff between the two configs
        all_attrs = set(orig_dynamic.config.keys()) | set(dynamic.config.keys())
        msg = ""
        for attr in all_attrs:
            orig_val = getattr(orig_dynamic.config, attr, None)
            chosen_val = getattr(dynamic.config, attr, None)
            if orig_val != chosen_val and not attr.startswith("_"):
                msg += f"'{attr}': {orig_val} -> {chosen_val}\n"
        if len(msg) != 0:
            logger.info("Diff between original -and> chosen dynamic:\n" + msg)
        else:
            logger.info("No config difference between original and loaded dynamic")

    else:
        logger.info(f"Loading dynamic from {run_path / 'saved_model' / 'dynamic'}")
        dynamic = orig_dynamic

    logger.info(f"Loaded dynamic:\n{dynamic}")

    ###############################################################################################
    #                                     Load Starting Images
    ###############################################################################################
    assert cfg.dataset.dataset_params is not None

    database_path = Path(cfg.dataset.path)
    logger.info(f"Using dataset {cfg.dataset.name} from {database_path}")
    subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    subdirs.sort(key=cfg.dataset.dataset_params.sorting_func)

    # some methos are only interested in the first subdir: timestep 0
    logger.info(f"Will be using subdir '{subdirs[0].name}' as starting point if applicable")

    # I reuse the Dataset class from the training script to load the images consistently,
    # but without augmentations to only start on truly true data
    starting_samples = list(subdirs[0].glob(f"*.{cfg.dataset.dataset_params.file_extension}"))
    kept_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)[0]
    starting_ds = cfg.dataset.dataset_params.dataset_class(
        starting_samples,
        kept_transforms,
        cfg.dataset.expected_initial_data_range,
    )
    logger.debug(f"Built dataset from {subdirs[0]}:\n{starting_ds}")
    logger.info(f"Using transforms:\n{kept_transforms}")

    ###############################################################################################
    #                                       Inference passes
    ###############################################################################################
    pbar_manager: Manager = get_manager()  # pyright: ignore[reportAssignmentType]

    eval_strats_msg = "\n".join([str(eval_strat) for eval_strat in cfg.evaluation_strategies])
    logger.info(f"Running {len(cfg.evaluation_strategies)} evaluation strategies:\n{eval_strats_msg}")

    for eval_strat_idx, eval_strat in enumerate(cfg.evaluation_strategies):
        logger.info(f"Running evaluation strategy {eval_strat_idx + 1}/{len(cfg.evaluation_strategies)}:\n{eval_strat}")
        logger.logger.name = eval_strat.name
        # add strategy-specific logger handler
        log_file_path = inference_conf.output_dir / eval_strat.name / "logs.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
        file_handler.setLevel(logging.DEBUG)
        logger.logger.addHandler(file_handler)
        # check for correct distributed environment
        is_distributed_strategy = isinstance(eval_strat, MetricsComputation)
        if not is_distributed_strategy and accelerator.distributed_type != DistributedType.NO:
            raise RuntimeError("Only MetricsComputation is supported in distributed mode")
        # run the evaluation strategy
        if type(eval_strat) is SimpleGeneration:
            simple_gen(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
            )
        elif type(eval_strat) is ForwardNoising:
            forward_noising(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
            )
        elif type(eval_strat) is ForwardNoisingLinearScaling:
            forward_noising_linear_scaling(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
            )
        elif type(eval_strat) is InvertedRegeneration:
            inverted_regeneration(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
            )
        elif type(eval_strat) is IterativeInvertedRegeneration:
            iterative_inverted_regeneration(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                logger,
            )
        elif type(eval_strat) is SimilarityWithTrainData:
            similarity_with_train_data(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                pbar_manager,
                subdirs,
                logger,
                cfg.dataset.data_shape,  # pyright: ignore[reportArgumentType]
            )
        elif type(eval_strat) is MetricsComputation:
            ### Evenly-spaced timesteps are assumed to be the actual empirical timesteps; must change this if it changes in training code
            # TODO: allow generating at non-empirical timesteps
            empirical_timesteps = get_evenly_spaced_timesteps(len(subdirs))
            logger.info(
                f"Empirical timesteps: {[round(ts, 3) for ts in empirical_timesteps]} from {len(subdirs)} subdirs"
            )
            timesteps2classnames: dict[float, str] = dict(zip(empirical_timesteps, [s.name for s in subdirs]))
            ### We need the full datasets/loaders for this method
            training_run_folder = cfg.output_dir.parent
            # use the original training config in <training_run_folder>/my_conf to know if the training was with unpaired data
            try:
                training_was_with_unpaired_data = get_training_boolean_value(
                    training_run_folder / "my_conf" / "my_training_conf.py", "unpaired_data"
                )
            except AttributeError as e:
                logger.warning(str(e))
                training_was_with_unpaired_data = False
            metrics_computation(
                cfg,
                eval_strat,
                net,
                video_time_encoder,
                dynamic,
                empirical_timesteps,
                timesteps2classnames,
                pbar_manager,
                logger,
                accelerator,
                training_was_with_unpaired_data,
            )
        elif type(eval_strat) is InversionRegenerationOnly:
            inversion_and_regeneration_only(cfg, eval_strat, net, video_time_encoder, dynamic, pbar_manager, logger)
        else:
            raise ValueError(f"Unknown evaluation strategy {eval_strat}")


def get_training_boolean_value(filepath: Path, key: str):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "training":
                    # Check if the value is a call to Training(...)
                    if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "Training":
                        for kw in node.value.keywords:
                            if kw.arg == key:
                                if isinstance(kw.value, ast.Constant):
                                    return kw.value.value
    # Default to False if key is not found
    raise AttributeError(f"Could not find attribute {key} in the `training=Training(...)` assignement in {filepath}")


def simple_gen(
    cfg: InferenceConfig,
    eval_strat: SimpleGeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
):
    """
    Just simple generations.

    `batch` is ignored!
    """
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        inpt = input(f"\nOutput directory {base_save_path} already exists.\nOverwrite? (y/[n]) ")
        if inpt.lower() == "y":
            shutil.rmtree(base_save_path)
        else:
            logger.info("Refusing to overwrite; exiting.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 2. Generate the time encodings
    eval_video_times = torch.rand(len(batch), device=cfg.device, dtype=cfg.dtype)
    eval_video_times = torch.sort(eval_video_times).values  # torch.sort it for better viz
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
        model_output: torch.Tensor = net.forward(
            image,
            t,
            encoder_hidden_states=random_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        gen_pbar.update()

    gen_pbar.close()

    save_grid_of_images_or_videos(
        image,
        base_save_path,
        "simple_generations",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )


def similarity_with_train_data(
    cfg: InferenceConfig,
    eval_strat: SimilarityWithTrainData,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    subdirs: list[Path],
    logger: MultiProcessAdapter,
    sample_shape: tuple[int, int, int],
):
    """
    Test model memorization by:

    - generating n samples (from random noise)
    - computing the closest (generated, true) pair by similarity, for each generated image
    - plotting the distribution of these n closest similarities
    - showing the p < n closest pairs

    `batch` is ignored!

    All computations are forcefully performed on fp32 for numerical precision.

    Similarities can be Euclidean cosine, L2, or both.
    """
    # Checks
    assert cfg.dataset.dataset_params is not None

    # -1. Prepare output directory & change models to fp16
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        inpt = input(f"\nOutput directory {base_save_path} already exists.\nOverwrite? (y/[n]) ")
        if inpt.lower() == "y":
            shutil.rmtree(base_save_path)
        else:
            logger.info("Refusing to overwrite; exiting.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    if cfg.dtype != torch.float32:
        logger.warning(
            "Switching to fp32 for this evaluation strategy as we need high numerical precision to avoid discretization artifacts in the histograms."
        )
        net = net.to(torch.float32)  # pyright: ignore[reportArgumentType]
        video_time_encoder = video_time_encoder.to(torch.float32)  # pyright: ignore[reportArgumentType]

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Setup the giant dataset of all datasets
    all_samples = list(subdirs[0].parent.rglob(f"*/*.{cfg.dataset.dataset_params.file_extension}"))
    kept_transforms, removed_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)
    all_times_ds: BaseDataset = cfg.dataset.dataset_params.dataset_class(
        all_samples,
        kept_transforms,  # no augs, they will be manually performed
        cfg.dataset.expected_initial_data_range,
    )
    logger.debug(f"Built all-times dataset from {subdirs[0].parent}:\n{all_times_ds}")

    # TODO: 2. Compute the average image of the dataset (to remove it afterwards)

    # 3. Generate samples and compute closest similarities
    num_full_batches, remaining = divmod(eval_strat.nb_generated_samples, eval_strat.batch_size)
    actual_bses = [eval_strat.batch_size] * num_full_batches + ([remaining] if remaining != 0 else [])

    batches_pbar = pbar_manager.counter(
        total=eval_strat.nb_generated_samples,
        position=1,
        desc=f"Generating samples (batch size: {eval_strat.batch_size})",
        leave=False,
    )
    batches_pbar.refresh()

    # instantiate similarities
    if isinstance(eval_strat.metrics, str):
        eval_strat.metrics = [eval_strat.metrics]
    metrics: dict[str, Callable] = dict.fromkeys(eval_strat.metrics)  # pyright: ignore[reportAssignmentType]
    for metric in eval_strat.metrics:
        if metric == "cosine":
            metrics[metric] = lambda x, y: CosineSimilarity(dim=1, eps=1e-9)(x, y)
        elif metric == "L2":
            metrics[metric] = PairwiseDistance(p=2, eps=1e-9)
        else:
            raise ValueError(f"Unsupported metric {metric}; expected 'cosine' or 'L2'")

    # get augmentation factor
    augmented_imgs = generate_all_augs(torch.randn((128, 128, 3)), removed_transforms)
    aug_factor = len(augmented_imgs)
    logger.debug(f"Augmentation factor: {aug_factor}")
    all_sims = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples, len(all_times_ds), aug_factor),
            float("NaN"),
            device=cfg.device,
            dtype=torch.float32,
        )
        for metric_name in metrics.keys()
    }

    BEST_VAL = {"cosine": 0, "L2": float("inf")}
    COMPARISON_OPERATORS = {
        "cosine": operator.gt,
        "L2": operator.lt,
    }
    MAX_OR_MIN = {
        "cosine": torch.maximum,
        "L2": torch.minimum,
    }
    worst_values = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples,),
            BEST_VAL[metric_name],
            device=cfg.device,
            dtype=torch.float32,
        )
        for metric_name in metrics.keys()
    }
    # closest_ds_idx_aug_idx[metric_name][i] = (closest_ds_idx, closest_aug_idx)
    closest_ds_idx_aug_idx = {
        metric_name: torch.full(
            (eval_strat.nb_generated_samples, 2),
            -1,
            device=cfg.device,
            dtype=torch.int64,
        )
        for metric_name in metrics.keys()
    }

    for batch_idx, bs in enumerate(actual_bses):
        start = batch_idx * eval_strat.batch_size
        end = start + bs

        # generate samples
        eval_video_times = torch.rand(bs, device=cfg.device, dtype=torch.float32)
        random_video_time_enc = video_time_encoder.forward(eval_video_times).unsqueeze(1)

        gen_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Generating samples" + " " * 17,
            leave=False,
        )
        gen_pbar.refresh()

        generated_images = torch.randn((bs, *sample_shape), dtype=torch.float32, device=cfg.device)

        for t in gen_pbar(inference_scheduler.timesteps):
            model_output: torch.Tensor = net.forward(
                generated_images,
                t,
                encoder_hidden_states=random_video_time_enc,
                return_dict=False,
            )[0]
            generated_images = inference_scheduler.step(model_output, int(t), generated_images, return_dict=False)[0]
        gen_pbar.close()

        all_ds_pbar = pbar_manager.counter(
            total=len(all_times_ds),
            position=2,
            desc="Comparing to all training examples ",
            leave=False,
        )
        all_ds_pbar.refresh()

        # compute cosine similarities overtorch.full (augmented) all-times dataset and report largest
        generated_images = generated_images.to(torch.float32)

        all_times_dl = DataLoader(
            all_times_ds,
            batch_size=1,  # batch size *must* be 1 here
            num_workers=2,
            pin_memory=True,
            prefetch_factor=3,
            pin_memory_device=cfg.device,
        )

        for img_idx, img in enumerate(all_ds_pbar(iter(all_times_dl))):
            assert len(img) == 1, f"Expected batch size 1, got {len(img)}"
            img = img[0].to(cfg.device)
            augmented_imgs = generate_all_augs(img, removed_transforms)  # also take into account the augmentations!

            for aug_img_idx, aug_img in enumerate(augmented_imgs):
                tiled_aug_img = aug_img.unsqueeze(0).tile(bs, 1, 1, 1)
                for metric_name, metric in metrics.items():
                    value = metric(tiled_aug_img.flatten(1), generated_images.flatten(1))
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

        all_ds_pbar.close()
        batches_pbar.update(bs)

        # plot closest pairs side-by-side
        if batch_idx < eval_strat.nb_batches_shown:
            for metric_name in metrics.keys():
                closest_true_imgs_idxes = closest_ds_idx_aug_idx[metric_name][start:end, 0].tolist()
                aug_idxes = closest_ds_idx_aug_idx[metric_name][start:end, 1]
                closest_true_imgs = all_times_ds.__getitems__(closest_true_imgs_idxes)
                closest_true_imgs_aug = torch.stack(
                    [
                        generate_all_augs(closest_true_imgs[i], transforms=removed_transforms)[aug_idxes[i]]
                        for i in range(end - start)
                    ]
                )
                plot_side_by_side_comparison(
                    generated_images,
                    closest_true_imgs_aug,
                    base_save_path,
                    "generated_images",
                    "closest_true_images",
                    metric_name,
                    ["-1_1 raw"],
                    eval_strat.n_rows_displayed,
                    logger,
                )
    batches_pbar.close()

    # report the largest similarities
    for metric_name in metrics.keys():
        logger.info(
            f"Worst found {metric_name} similarities: {[round(val, 3) for val in worst_values[metric_name].tolist()]}"
        )
        closest_true_imgs_names = [
            Path(all_times_ds.samples[idx]).name for idx in closest_ds_idx_aug_idx[metric_name][:, 0]
        ]
        logger.debug(f"Closest found images: {closest_true_imgs_names}")

    # torch.save all metrics and plot their histogram
    for metric_name in metrics.keys():
        this_metric_all_sims = all_sims[metric_name].cpu()
        if torch.any(torch.isnan(this_metric_all_sims)):
            logger.warning("Found NaNs in {metric_name} similarities")
        torch.save(this_metric_all_sims, base_save_path / f"all_{metric_name}.pt")
        plt.figure(figsize=(10, 6))
        plt.hist(this_metric_all_sims.flatten().numpy(), bins=300)
        plt.title(
            f"nb_samples_generated × nb_train_samples × augment_factor = {eval_strat.nb_generated_samples} × {len(all_times_ds)} × {aug_factor} = {this_metric_all_sims.numel():,}"
        )
        plt.suptitle(f"Distribution of all {metric_name} similarities")
        plt.grid()
        plt.tight_layout()
        plt.savefig(base_save_path / f"worst_{metric_name}_hist.png")

    # plot the histogram of each per-generated-image worst similarity, for each metric
    for metric_name in metrics.keys():
        plt.figure(figsize=(10, 6))
        plt.hist(worst_values[metric_name].flatten().cpu().numpy(), bins=300)
        plt.title(f"nb_samples_generated = {eval_strat.nb_generated_samples} = {worst_values[metric_name].numel():,}")
        plt.suptitle(f"Distribution of all {metric_name} *worst* similarities")
        plt.grid()
        plt.tight_layout()
        plt.savefig(base_save_path / f"all_{metric_name}_hist.png")


def forward_noising(
    cfg: InferenceConfig,
    eval_strat: ForwardNoising,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup scheduler
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 0.5. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
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
    noise_timesteps: torch.IntTensor = torch.full(  # pyright: ignore[reportAssignmentType]
        (batch.shape[0],),
        noise_timestep,
        device=batch.device,
        dtype=torch.int64,
    )
    slightly_noised_sample = inference_scheduler.add_noise(batch, noise, noise_timesteps)
    save_grid_of_images_or_videos(
        slightly_noised_sample,
        base_save_path,
        "noised_samples",
        ["image min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # usefull if small inference batch size only
    video = []
    nb_vid_batches = ceil(eval_strat.nb_video_timesteps / eval_strat.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * eval_strat.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * eval_strat.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        logger.debug(f"Processing video times from {start} to {start + len(video_time_batch)}")
        # at this point video_time_batch is at most eval_strat.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the actual batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        video_time_enc = video_time_enc.repeat_interleave(eval_strat.nb_generated_samples, dim=0)

        image = torch.cat([slightly_noised_sample.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # torch.where len(video_time_batch) = eval_strat.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
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
    video = video.split(eval_strat.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )


def forward_noising_linear_scaling(
    cfg: InferenceConfig,
    eval_strat: ForwardNoisingLinearScaling,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
):
    # -2. Checks
    if eval_strat.nb_video_times_in_parallel != 1:
        logger.warning(
            f"nb_video_times_in_parallel was set to {eval_strat.nb_video_times_in_parallel}, but is ignored in this evaluation strategy"
        )

    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        raise FileExistsError(f"Output directory {base_save_path} already exists. Refusing to overwrite.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup scheduler
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 0.5. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )

    # 2. Sample Gaussian noise
    noise = torch.randn_like(batch)

    # 2.5 Misc preparations
    video = []
    video_time_pbar = pbar_manager.counter(
        total=eval_strat.nb_video_timesteps,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    # video times between 0 and 1
    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)
    # linearly interpolate between the start and end forward_noising_fracs
    forward_noising_fracs = torch.linspace(
        eval_strat.forward_noising_frac_start,
        eval_strat.forward_noising_frac_end,
        eval_strat.nb_video_timesteps,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    logger.debug(f"Forward noising fracs: {list(forward_noising_fracs)}")

    # 3. Generate the trajectory time-per-time
    for vid_batch_idx in range(eval_strat.nb_video_timesteps):
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
        noise_timesteps: torch.IntTensor = torch.full(  # pyright: ignore[reportAssignmentType]
            (batch.shape[0],),
            noise_timestep,
            device=batch.device,
            dtype=torch.int64,
        )
        image = inference_scheduler.add_noise(batch, noise, noise_timesteps)  # clone happens here
        # shape: (batch_size, channels, height, width)

        # denoise with backward SDE
        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
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
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )


def inversion_and_regeneration_only(
    cfg: InferenceConfig,
    eval_strat: InversionRegenerationOnly,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        inpt = input(f"\nOutput directory {base_save_path} already exists.\nOverwrite? (y/[n]) ")
        if inpt.lower() == "y":
            shutil.rmtree(base_save_path)
        else:
            logger.info("Refusing to overwrite; exiting.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 0.5. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
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
            inverted_gauss,
            t,
            encoder_hidden_states=inversion_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        inverted_gauss.to(torch.float32),  # needed for the 5% - 95% (numpy) normalization
        base_save_path,
        "inverted_gaussians",
        ["image min-max", "image 5perc-95perc"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )

    # 3. Regenerate the original starting images from it
    diff_time_pbar = pbar_manager.counter(
        total=len(inference_scheduler.timesteps),
        position=1,
        desc="Diffusion timesteps" + " " * 4,
        leave=False,
    )
    diff_time_pbar.refresh()

    video_time_enc = inversion_video_time_enc  # same time!
    image = inverted_gauss.clone()

    for t in inference_scheduler.timesteps:
        model_output: torch.Tensor = net.forward(
            image,
            t,
            encoder_hidden_states=video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        image,
        base_save_path,
        "regeneration",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )


def inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
):
    # -1. Prepare output directory
    base_save_path = cfg.output_dir / eval_strat.name
    if base_save_path.exists():
        inpt = input(f"\nOutput directory {base_save_path} already exists.\nOverwrite? (y/[n]) ")
        if inpt.lower() == "y":
            shutil.rmtree(base_save_path)
        else:
            logger.info("Refusing to overwrite; exiting.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 0.5. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
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
            inverted_gauss,
            t,
            encoder_hidden_states=inversion_video_time_enc.unsqueeze(1),
            return_dict=False,
        )[0]
        inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
        diff_time_pbar.update()

    diff_time_pbar.close()

    save_grid_of_images_or_videos(
        inverted_gauss.to(torch.float32),  # needed for the 5% - 95% (numpy) normalization
        base_save_path,
        "inverted_gaussians",
        ["image min-max", "image 5perc-95perc"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # useful if small inference batch size only
    video = []
    nb_vid_batches = ceil(eval_strat.nb_video_timesteps / eval_strat.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * eval_strat.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * eval_strat.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        logger.debug(f"Processing video times from {start} to {start + len(video_time_batch)}")
        # at this point video_time_batch is at most cfg.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the actual batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        video_time_enc = video_time_enc.repeat_interleave(eval_strat.nb_generated_samples, dim=0)

        image = torch.cat([inverted_gauss.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # torch.where len(video_time_batch) = cfg.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps:
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
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
    video = video.split(eval_strat.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    # 2 separate calls here to only save the individual frames of the -1_1 raw version
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["-1_1 raw"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
        also_save_individual_frames=True,
    )


def iterative_inverted_regeneration(
    cfg: InferenceConfig,
    eval_strat: IterativeInvertedRegeneration,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
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
        inpt = input(f"\nOutput directory {base_save_path} already exists.\nOverwrite? (y/[n]) ")
        if inpt.lower() == "y":
            shutil.rmtree(base_save_path)
        else:
            logger.info("Refusing to overwrite; exiting.")
    base_save_path.mkdir(parents=True)
    logger.debug(f"Saving outputs to {base_save_path}")

    # 0. Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    # thresholding is not supported by DDIMInverseScheduler; use "raw" clipping instead
    if dynamic.config["thresholding"]:
        kwargs = {"clip_sample": True, "clip_sample_range": 1}
    else:
        kwargs = {}
    inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(dynamic.config, **kwargs)  # pyright: ignore[reportAssignmentType]
    inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
    logger.debug(f"Using inverted scheduler config: {inverted_scheduler.config}")

    # 0.5. Get the starting batch
    batch = get_starting_batch(cfg, eval_strat, logger, cfg.dataset.path, cfg.device, cfg.dtype)

    # 1. Save the to-be noised images
    save_grid_of_images_or_videos(
        batch,
        base_save_path,
        "starting_samples",
        ["-1_1 raw", "image min-max"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )
    save_histogram(batch, base_save_path / "starting_samples_histogram.png", (-1, 1), 50)

    # 2. Generate the trajectory
    video = []

    video_time_pbar = pbar_manager.counter(
        total=eval_strat.nb_video_timesteps,
        position=1,
        desc="Video timesteps",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, eval_strat.nb_video_timesteps, device=cfg.device, dtype=cfg.dtype)

    prev_video_time = 0
    image = batch
    for video_t_idx, video_time in enumerate(video_times):
        logger.debug(f"Video timestep index {video_t_idx + 1} / {eval_strat.nb_video_timesteps}")

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
                inverted_gauss,
                t,
                encoder_hidden_states=inversion_video_time.unsqueeze(1),
                return_dict=False,
            )[0]
            inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        save_grid_of_images_or_videos(
            inverted_gauss,
            base_save_path,
            f"inverted_gaussians_time{video_t_idx}",
            ["image min-max"],
            eval_strat.n_rows_displayed,
            0 if eval_strat.plate_name_to_simulate is not None else 2,
            logger,
        )
        save_histogram(
            inverted_gauss,
            base_save_path / f"inverted_gaussians_histogram_time{video_t_idx}.png",
            (-5, 5),
        )

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
            model_output: torch.Tensor = net.forward(
                image,
                t,
                encoder_hidden_states=video_time_enc.unsqueeze(1),
                return_dict=False,
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()

        video.append(image.clone())
        save_histogram(
            image,
            base_save_path / f"image_histogram_time{video_t_idx}.png",
            (-1, 1),
            50,
        )

        prev_video_time = video_time.item()
        video_time_pbar.update()

    video_time_pbar.close(clear=True)

    # video is a list of nb_video_timesteps elements, each of shape (batch_size, channels, hight, width)
    video = torch.stack(video)  # (nb_video_timesteps, batch_size, channels, height, width)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        eval_strat.nb_video_timesteps,
        eval_strat.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    logger.debug(f"Saving video tensor of shape {video.shape}")
    save_grid_of_images_or_videos(
        video,
        base_save_path,
        "trajectories",
        ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
        eval_strat.n_rows_displayed,
        0 if eval_strat.plate_name_to_simulate is not None else 2,
        logger,
    )


def metrics_computation(
    cfg: InferenceConfig,
    eval_strat: MetricsComputation,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    timesteps: list[float],
    timesteps2classnames: dict[float, str],
    pbar_manager: Manager,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
    training_was_with_unpaired_data: bool,
):
    """
    Compute metrics such as FID.

    Everything is distributed.

    Adapted from utils/training.py
    """
    ##### 0. Preparations
    # Setup schedulers
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # pyright: ignore[reportAssignmentType]
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # Misc.
    logger.info(f"Starting {eval_strat.name}: {eval_strat}")
    logger.debug(
        f"Starting {eval_strat.name} on process ({accelerator.process_index})",
        main_process_only=False,
    )
    metrics_computation_folder = cfg.output_dir / eval_strat.name
    existing_problematic_files = [f for f in metrics_computation_folder.iterdir() if f.name != "logs.log"]
    if accelerator.is_main_process and not len(existing_problematic_files) == 0:
        logger.warning(
            f"Output directory {metrics_computation_folder.name} already exists and is not empty: {[f.name for f in existing_problematic_files]}"
        )
        inpt = input("Overwrite these files/folders? Will exist otherwise (y/[n]) ")
        if inpt.lower() != "y":
            raise RuntimeError("Refusing to proceed with a non-empty folder.")
        else:
            accelerator.print(f"Deleting files/folders:{existing_problematic_files}")
            for f in existing_problematic_files:
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()
            accelerator.print("Recreating the folder")
    accelerator.wait_for_everyone()

    # use training time encodings
    assert list(timesteps2classnames.keys()) == timesteps, (
        f"Expected timesteps to be the same as keys in timesteps2classnames, got {timesteps} and {timesteps2classnames.keys()}"
    )
    classnames2timesteps = {v: k for k, v in timesteps2classnames.items()}
    eval_video_times = [classnames2timesteps[str(eval_time)] for eval_time in eval_strat.selected_times]
    logger.info(f"Selected video times for metrics computation: {eval_video_times} from {eval_strat.selected_times}")
    eval_video_time_enc = video_time_encoder.forward(torch.tensor(eval_video_times).to(accelerator.device, cfg.dtype))

    # get true datasets to compare against (in [0; 255] uint8 PNG images)
    assert cfg.dataset.dataset_params is not None
    true_datasets_to_compare_with = get_true_datasets_for_metrics_computation(
        eval_strat,
        eval_video_times,
        cfg.dataset.dataset_params.dataset_class,
        cfg.dataset.dataset_params.file_extension,
        cfg.dataset.transforms,
        cfg,
        logger,
        timesteps2classnames,
        training_was_with_unpaired_data,
        metrics_computation_folder,
    )

    ##### 1. Generate the samples
    if eval_strat.regen_images:
        _generate_images_for_metrics_computation(
            pbar_manager,
            eval_video_times,
            accelerator,
            eval_video_time_enc,
            eval_strat,
            timesteps2classnames,
            true_datasets_to_compare_with,
            metrics_computation_folder,
            training_was_with_unpaired_data,
            inference_scheduler,
            net,
            cfg,
            logger,
        )
    else:
        logger.warning("Skipping image generation for metrics computation")

    ##### 2. Compute metrics
    # TODO: weight tasks by number of samples
    # TODO: include "all_times" in the tasks, but ***differentiate between using seen data only and all the available dataset!***
    # tasks = ["all_times"] + list(timesteps_names_to_floats.keys())
    tasks = [timesteps2classnames[eval_time] for eval_time in eval_video_times]  # eval time names
    tasks_for_this_process = tasks[accelerator.process_index :: accelerator.num_processes]
    logger.debug(f"Tasks for process {accelerator.process_index}: {tasks_for_this_process}", main_process_only=False)

    logger.info("Computing metrics...")
    metrics_dict: dict[str, dict[str, float]] = {}
    for task in tasks_for_this_process:
        gen_samples_input = metrics_computation_folder / task if task != "all_times" else metrics_computation_folder
        nb_samples_gen = len(list(gen_samples_input.iterdir()))
        nb_true_samples = len(true_datasets_to_compare_with[task])
        if nb_samples_gen != nb_true_samples:
            logger.warning(
                f"Mismatch in the number of samples for task {task}: {nb_samples_gen} generated vs {nb_true_samples} true"
            )
        logger.debug(
            f"Computing metrics on {nb_samples_gen} generated samples at {gen_samples_input.as_posix()} vs {nb_true_samples} true samples at {true_datasets_to_compare_with[task].base_path} on process {accelerator.process_index}",
            main_process_only=False,
        )
        # no caching as the true dataset is fixed only in case `nb_samples_to_gen_per_time` is a number or 'adapt',
        # so we should use caching only in that case => never cache for now
        metrics = torch_fidelity.calculate_metrics(
            input1=true_datasets_to_compare_with[task],
            input2=gen_samples_input.as_posix(),
            cuda=True,
            batch_size=eval_strat.batch_size * 2,  # TODO: optimize
            isc=True,
            fid=True,
            prc=True,
            verbose=accelerator.is_main_process,
            cache=False,
            samples_find_deep=task == "all_times",
        )
        metrics_dict[task] = metrics
        logger.debug(
            f"Computed metrics for time {task} on process {accelerator.process_index}: {metrics}",
            main_process_only=False,
        )
    # save this process' metrics to disk
    with open(
        metrics_computation_folder / f"proc{accelerator.process_index}_metrics_dict.pkl",
        "wb",
    ) as pickle_file:
        pickle.dump(metrics_dict, pickle_file)
        logger.debug(
            f"Saved metrics for process {accelerator.process_index} at {metrics_computation_folder}",
            main_process_only=False,
        )
    accelerator.wait_for_everyone()

    ##### 3. Merge metrics from all processes & save them
    final_metrics_dict: dict[str, dict[str, float]] = {}
    # load all processes' metrics
    for metrics_file in [f for f in metrics_computation_folder.iterdir() if f.name.endswith("metrics_dict.pkl")]:
        with open(metrics_file, "rb") as pickle_file:
            proc_metrics = pickle.load(pickle_file)
            final_metrics_dict.update(proc_metrics)
    # save it on main process
    if accelerator.is_main_process:
        with open(
            metrics_computation_folder / "all_procs_metrics_dict.pkl",
            "wb",
        ) as pickle_file:
            pickle.dump(final_metrics_dict, pickle_file)
    logger.info(
        f"Saved metrics: {final_metrics_dict} to {metrics_computation_folder}/all_procs_metrics_dict.pkl",
    )


def _generate_images_for_metrics_computation(
    pbar_manager: Manager,
    eval_video_times: list[float],
    accelerator: Accelerator,
    eval_video_time_enc: Tensor,
    eval_strat: MetricsComputation,
    timesteps2classnames: dict[float, str],
    true_datasets_to_compare_with: dict[str, BaseDataset],
    metrics_computation_folder: Path,
    training_was_with_unpaired_data: bool,
    inference_scheduler: DDIMScheduler,
    net: UNet2DConditionModel,
    cfg: InferenceConfig,
    logger: MultiProcessAdapter,
):
    # loop over training video times
    video_times_pbar = pbar_manager.counter(
        total=len(eval_video_times),
        position=2,
        desc="Training video timesteps  ",
        enable=accelerator.is_main_process,
        leave=False,
    )
    if accelerator.is_main_process:
        video_times_pbar.refresh()

    for video_time_idx, video_time_enc in video_times_pbar(enumerate(eval_video_time_enc)):
        video_time_enc = video_time_enc.unsqueeze(0).repeat(eval_strat.batch_size, 1)

        # get timestep name
        video_time_name = timesteps2classnames[eval_video_times[video_time_idx]]

        # find how many samples to generate, batchify generation and distribute along processes
        gen_dir = metrics_computation_folder / video_time_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        this_proc_gen_batches = find_this_proc_this_time_batches_for_metrics_comp(
            eval_strat,
            true_datasets_to_compare_with[video_time_name].base_path / video_time_name,
            gen_dir,
            logger,
            training_was_with_unpaired_data,
            accelerator,
        )

        batches_pbar = pbar_manager.counter(
            total=len(this_proc_gen_batches),
            position=3,
            desc="Evaluation batch" + 10 * " ",
            enable=accelerator.is_main_process,
            leave=False,
        )
        if accelerator.is_main_process:
            batches_pbar.refresh()

        # loop over generation batches
        for batch_size in batches_pbar(this_proc_gen_batches):
            gen_pbar = pbar_manager.counter(
                total=len(inference_scheduler.timesteps),
                position=4,
                desc="Generating samples" + " " * 8,
                enable=accelerator.is_main_process,
                leave=False,
            )
            if accelerator.is_main_process:
                gen_pbar.refresh()

            # generate a batch of samples
            image = torch.randn(
                batch_size,
                accelerator.unwrap_model(net).config["in_channels"],
                accelerator.unwrap_model(net).config["sample_size"],
                accelerator.unwrap_model(net).config["sample_size"],
                device=accelerator.device,
                dtype=cfg.dtype,
            )
            video_time_enc = video_time_enc[:batch_size]

            # loop over diffusion timesteps
            for t in gen_pbar(inference_scheduler.timesteps):
                model_output: torch.Tensor = net.forward(image, t, video_time_enc.unsqueeze(1), return_dict=False)[0]  # pyright: ignore[reportArgumentType]
                image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

            # save to [0; 255] uint8 PNG images
            save_images_for_metrics_compute(
                image,
                gen_dir,
                accelerator.process_index,
            )

            gen_pbar.close(clear=True)

        batches_pbar.close(clear=True)
        # wait for everyone at end of each time (should be enough to avoid timeouts)
        accelerator.wait_for_everyone()

    video_times_pbar.close(clear=True)
    # no need to wait here then
    logger.debug("Finished image generation in MetricsComputation")

    ##### 1.5 Augment the generated samples if applicable
    if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
        logger.info("Augmenting generated samples for metrics computation")
        # Partition generated subdirectories among processes
        subdirs = [metrics_computation_folder / video_time_name for video_time_name in timesteps2classnames.values()]
        assigned_subdirs = subdirs[accelerator.process_index :: accelerator.num_processes]
        n_workers_per_process = os.cpu_count() // accelerator.num_processes
        for subdir in assigned_subdirs:
            logger.debug(f"Process {accelerator.process_index} augmenting folder {subdir}", main_process_only=False)
            # augment
            extension = "png"  # TODO: remove this hardcoded extension (by moving DatasetParams'params into the base DataSet class used in config, another TODO)
            hard_augment_dataset_all_square_symmetries(subdir, logger, extension, n_workers_per_process)
            # check result
            assert (nb_elems := len(list((subdir).glob(f"*.{extension}"))) % 8 == 0), (
                f"Expected number of elements to be a multiple of 8, got:\n{nb_elems} in {subdir}"
            )

    # wait for data augmentation to finish before returning
    accelerator.wait_for_everyone()


def save_grid_of_images_or_videos(
    tensor: torch.Tensor,
    base_save_path: Path,
    artifact_name: str,
    norm_methods: list[str],
    nrows: int,
    padding: int,
    logger: MultiProcessAdapter,
    also_save_individual_frames: bool = False,
):
    """Save a `tensor` of images or videos to disk in a grid of `nrows`×`nrows`."""
    # Save some raw images / trajectories to disk
    file_path = base_save_path / f"{artifact_name}.pt"
    torch.save(tensor.half().cpu(), file_path)
    logger.debug(f"Saved raw {artifact_name} of shape {tensor.shape} to {file_path.name}")

    normalized_elements = normalize_elements_for_logging(tensor, norm_methods)

    match tensor.ndim:
        case 5:  # videos
            for norm_method, normed_vids in normalized_elements.items():
                # torch.save the videos in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.mp4"
                _save_grid_of_videos(normed_vids, save_path, nrows, padding, logger, also_save_individual_frames)
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")
        case 4:  # images
            for norm_method, normed_imgs in normalized_elements.items():
                # torch.save the images in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.png"
                _save_grid_of_images(normed_imgs, save_path, nrows, padding, logger)
                logger.debug(f"Saved {norm_method} normalized {artifact_name} to {save_path.name}")
        case _:
            raise ValueError(f"Expected 4D or 5D tensor, got {tensor.ndim} with shape {tensor.shape}")

    return save_path  # pyright: ignore[reportPossiblyUnboundVariable]


def _save_grid_of_videos(
    videos_tensor: ndarray,
    save_path: Path,
    nrows: int,
    padding: int,
    logger: MultiProcessAdapter,
    also_save_individual_frames: bool = False,
):
    # Checks
    assert videos_tensor.ndim == 5, f"Expected 5D tensor, got {videos_tensor.shape}"
    assert videos_tensor.dtype == np.uint8, f"Expected dtype uint8, got {videos_tensor.dtype}"
    assert videos_tensor.min() >= 0 and videos_tensor.max() <= 255, (
        f"Expected [0;255] range, got [{videos_tensor.min()}, {videos_tensor.max()}]"
    )
    if videos_tensor.shape[1] != nrows**2:
        logger.warning(
            f"Expected nrows²={nrows**2} videos at dim index 1, got shape: {videos_tensor.shape}. Selecting first {nrows**2} videos."
        )
        videos_tensor = videos_tensor[:, : nrows**2]

    # Convert tensor to a grid of videos
    fps = max(1, int(len(videos_tensor) / 10))
    logger.debug(f"Using fps {fps}")
    writer = imageio.get_writer(save_path, mode="I", fps=fps)

    if also_save_individual_frames:
        (save_path.parent / save_path.stem).mkdir(exist_ok=True)

    for frame_idx, frame in enumerate(videos_tensor):
        grid_img = make_grid(torch.from_numpy(frame), nrow=nrows, padding=padding)
        pil_img = grid_img.numpy().transpose(1, 2, 0)
        logger.debug(f"Adding frame {frame_idx + 1}/{len(videos_tensor)} | shape: {pil_img.shape}")
        writer.append_data(pil_img)
        if also_save_individual_frames:
            frame_path = save_path.parent / save_path.stem / f"frame_{frame_idx}.tiff"
            Image.fromarray(pil_img).save(frame_path)
            logger.debug(f"Saved frame {frame_idx + 1}/{len(videos_tensor)} at {frame_path}")

    writer.close()


def _save_grid_of_images(
    images_tensor: ndarray, save_path: Path, nrows: int, padding: int, logger: MultiProcessAdapter
):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.dtype == np.uint8, f"Expected dtype uint8, got {images_tensor.dtype}"
    assert images_tensor.min() >= 0 and images_tensor.max() <= 255, (
        f"Expected [0;255] range, got [{images_tensor.min()}, {images_tensor.max()}]"
    )
    if images_tensor.shape[0] != nrows**2:
        logger.warning(
            f"Expected nrows²={nrows**2} images, got {images_tensor.shape[0]}. Selecting first {nrows**2} images."
        )
        images_tensor = images_tensor[: nrows**2]

    # Convert tensor to a grid of images
    grid_img = make_grid(torch.from_numpy(images_tensor), nrow=nrows, padding=padding)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img.numpy().transpose(1, 2, 0))
    pil_img.save(save_path)


def save_histogram(
    images_tensor: torch.Tensor,
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
        plt.plot(
            x,
            stats.norm.pdf(x),
            color="grey",
            linestyle="dashed",
            label="Gaussian",
            alpha=0.6,
        )

    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram to {save_path}")


def plot_side_by_side_comparison(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    base_save_path: Path,
    artifact1_name: str,
    artifact2_name: str,
    metric_name: str,
    norm_methods: list[str],
    nrows: int,
    logger: MultiProcessAdapter,
):
    """
    Save a side-by-side comparison of two tensors of images or videos to disk in a grid of `nrows`×`nrows`,
    with 2 images / videos per cell
    """
    # Checks
    assert tensor1.shape == tensor2.shape, f"Expected same shape, got {tensor1.shape} and {tensor2.shape}"

    # Save some raw images / trajectories to disk
    base_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor1.half().cpu(), base_save_path / f"{artifact1_name}.pt")
    logger.debug(f"Saved raw {artifact1_name} of shape {tensor1.shape} to {base_save_path}/{artifact1_name}.pt")
    torch.save(tensor2.half().cpu(), base_save_path / f"{artifact2_name}.pt")
    logger.debug(f"Saved raw {artifact2_name} of shape {tensor2.shape} to {base_save_path}/{artifact2_name}.pt")

    normalized_t1 = normalize_elements_for_logging(tensor1, norm_methods)
    normalized_t2 = normalize_elements_for_logging(tensor2, norm_methods)

    match tensor1.ndim:
        case 5:  # videos
            raise NotImplementedError("TODO if needed")
        case 4:  # images
            for norm_method in normalized_t1.keys():
                t1, t2 = normalized_t1[norm_method], normalized_t2[norm_method]
                # torch.save the images in a grid
                save_path = base_save_path / f"{metric_name}_{artifact1_name}_vs_{artifact2_name}_{norm_method}.png"
                _save_side_by_side_of_images(t1, t2, save_path, nrows, logger)
                logger.debug(
                    f"Saved {norm_method} normalized {artifact1_name} vs {artifact2_name} side-by-side to {save_path.name}"
                )
        case _:
            raise ValueError(f"Expected 4D or 5D tensor, got {tensor1.ndim} with shape {tensor1.shape}")

    return save_path  # pyright: ignore[reportPossiblyUnboundVariable]


def _save_side_by_side_of_images(
    t1: ndarray,
    t2: ndarray,
    save_path: Path,
    nrows: int,
    logger: MultiProcessAdapter,
):
    # Checks
    assert t1.ndim == t2.ndim == 4, f"Expected 4D tensor, got {t1.shape} and {t2.shape}"
    assert t1.shape == t2.shape, f"Expected same shape, got {t1.shape} and {t2.shape}"
    assert t1.dtype == t2.dtype == np.uint8, f"Expected dtype uint8, got {t1.dtype} and {t2.dtype}"
    assert t1.min() >= 0 and t1.max() <= 255, f"Expected [0;255] range, got [{t1.min()}, {t1.max()}]"
    assert t2.min() >= 0 and t2.max() <= 255, f"Expected [0;255] range, got [{t2.min()}, {t2.max()}]"
    if t1.shape[0] != nrows**2:
        logger.warning(f"Expected nrows²={nrows**2} images, got {t1.shape[0]}. Selecting first {nrows**2} images.")
        t1 = t1[: nrows**2]
        t2 = t2[: nrows**2]

    # interleave the two tensors
    interleaved_imgs = np.empty((t1.shape[0] * 2, *t1.shape[1:]), dtype=t1.dtype)
    interleaved_imgs[0::2] = t1
    interleaved_imgs[1::2] = t2

    # Convert tensor to a grid of images
    grid_img = make_grid(torch.from_numpy(interleaved_imgs), nrow=nrows * 2)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img.numpy().transpose(1, 2, 0))
    pil_img.save(save_path)


def get_true_datasets_for_metrics_computation(
    eval_strat: MetricsComputation,
    eval_video_times: list[float],
    dataset_class: type[BaseDataset],
    data_files_common_suffix: str,
    test_transforms: Compose,
    cfg: InferenceConfig,
    logger: MultiProcessAdapter,
    timesteps2classnames: dict[float, str],
    training_was_with_unpaired_data: bool,
    metrics_computation_folder: Path,
):
    """
    Return the true datasets to compare against for metrics computation, in [0; 255] uint8 tensors like saved generated images.

    Depending on `eval_strat.nb_samples_to_gen_per_time` the true datasets returned by this function can be:
    - the hard augmented versions if `nb_samples_to_gen_per_time == "adapt half aug"`
    - half of the whatever is used from last step if `"half" in nb_samples_to_gen_per_time`

    Copied/Adapted from `GaussianProxy/utils/training.py`.
    """
    # Misc.
    base_dataset_path = Path(cfg.dataset.path)
    eval_time_names = [timesteps2classnames[eval_time] for eval_time in eval_video_times]
    true_datasets_to_compare_with: dict[str, BaseDataset] = {}

    # Use the correct image processing ([0; 255] uint8) for metrics computation
    assert any(isinstance(t, Normalize) for t in test_transforms.transforms), (
        f"Expected normalization to be in test transforms, got : {test_transforms}"
    )
    nb_channels = cfg.dataset.data_shape[0]
    metrics_compute_transforms = Compose(
        [
            test_transforms,  # test transforms *must* include the normalization to [-1, 1]
            Normalize(mean=[-1] * nb_channels, std=[2] * nb_channels),  # Convert from [-1, 1] to [0, 1],
            ConvertImageDtype(torch.uint8),  # this also scales to [0, 255]
        ]
    )
    logger.info(f"Using transforms for metrics computation: {metrics_compute_transforms}")
    # TODO: programmatically check consistency with true samples processing in misc/save_images_for_metrics_compute

    # Force using the hard augmented version of the dataset if generating augmented samples
    all_files_per_time = {}
    if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
        ds_path = base_dataset_path.with_name(base_dataset_path.name.strip("_hard_augmented") + "_hard_augmented")
        assert ds_path.exists(), f"Expected existing hard augmented dataset at {ds_path}"
        logger.debug(f"Using hard augmented version at {ds_path}")
    # Otherwise, simply use all available data from the dataset we're using to train (and test)
    else:
        ds_path = base_dataset_path

    # Then retrieve all files for each time
    def list_files(time_name: str):
        current_path = ds_path / time_name
        files = [f for f in current_path.iterdir() if f.is_file() and f.suffix == "." + data_files_common_suffix]
        return time_name, files

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(list_files, t): t for t in eval_time_names}
        for future in as_completed(futures):
            time_name, files = future.result()
            all_files_per_time[time_name] = files

    # Then if generating half the number of samples, take half of the available true data to compare with
    for time_name in all_files_per_time.keys():
        all_files_per_time[time_name] = all_files_per_time[time_name][: len(all_files_per_time[time_name]) // 2]

    # Finally instantiate the datasets
    for time_name, all_files in all_files_per_time.items():
        true_datasets_to_compare_with[time_name] = dataset_class(
            samples=all_files,
            transforms=metrics_compute_transforms,
        )

    # Check that datasets are well-formed
    for time_name, dataset in true_datasets_to_compare_with.items():
        assert len(dataset) > 0, (
            f"No samples found for time {time_name} when creating true dataset to compare with for metrics computation"
        )
        if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
            assert training_was_with_unpaired_data or len(dataset) % 8 == 0, (
                f"Expected number of samples to be a multiple of 8 when using paired data, got {len(dataset)}"
            )
        logger.debug(
            f"True dataset to compare with for metrics computation for time {time_name} has {len(dataset)} samples at {dataset.base_path}"
        )

    # Save a few samples to visually check the processing!
    logger.info(
        f"Saving a few true samples for processing check at {metrics_computation_folder}/few_true_samples_for_processing_check"
    )
    for time, dataset in true_datasets_to_compare_with.items():
        few_samples_indexes = random.sample(range(len(dataset)), 5)
        few_samples = dataset.__getitems__(few_samples_indexes)
        for sample_idx, sample in enumerate(few_samples):
            pil_img = Image.fromarray(sample.permute(1, 2, 0).numpy())
            orig_filename = dataset.samples[few_samples_indexes[sample_idx]].name
            out_dir = metrics_computation_folder / "few_true_samples_for_processing_check" / time
            out_dir.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_dir / f"{orig_filename}_processed.png")

    return true_datasets_to_compare_with


def find_this_proc_this_time_batches_for_metrics_comp(
    eval_strat: MetricsComputation,
    true_data_class_path: Path,
    gen_dir: Path,
    logger: MultiProcessAdapter,
    training_was_with_unpaired_data: bool,
    accelerator: Accelerator,
) -> list[int]:
    """
    Return the list of batch sizes to generate, for a given `video_time_idx` and splitting between processes.

    `eval_strat.nb_samples_to_gen_per_time` can be:
    - an `int`: the number of samples to generate
    - `"adapt"`: generate as many samples as there are in the true data time
    - `"adapt half"`: generate half as many samples as there are in the true data time
    - `"adapt half aug"`: generate half as many samples as there are in the true data time, then 8⨉ augment them (Dih4)

    Copied/Adapted from `GaussianProxy/utils/training.py`.
    """
    # find total number of samples to generate for this video time
    if isinstance(eval_strat.nb_samples_to_gen_per_time, int):
        tot_nb_samples = eval_strat.nb_samples_to_gen_per_time
    elif eval_strat.nb_samples_to_gen_per_time == "adapt":
        tot_nb_samples = len(list(true_data_class_path.iterdir()))
    elif eval_strat.nb_samples_to_gen_per_time == "adapt half":
        tot_nb_samples = len(list(true_data_class_path.iterdir())) // 2
    elif eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
        nb_all_aug_samples = len(list(true_data_class_path.iterdir()))
        assert training_was_with_unpaired_data or nb_all_aug_samples % 8 == 0, (
            f"Expected number of samples to be a multiple of 8 when using paired data, got {nb_all_aug_samples}"
        )
        tot_nb_samples = nb_all_aug_samples // (8 * 2)  # half the number of samples, *then* 8⨉ augment them
        logger.debug("Will augment samples 8⨉ after generation")
    else:
        raise ValueError(
            f"Expected 'nb_samples_to_gen_per_time' to be an int, 'adapt', 'adapt half', or 'adapt half aug', got {eval_strat.nb_samples_to_gen_per_time}"
        )
    logger.debug(
        f"Will generate {tot_nb_samples} samples for time {true_data_class_path.name} at {true_data_class_path.as_posix()}"
    )

    # Resume from interrupted generation
    already_existing_samples = len(list(gen_dir.iterdir()))
    logger.debug(
        f"Found {already_existing_samples} already existing samples for time {true_data_class_path.name} at {gen_dir}"
    )
    tot_nb_samples -= already_existing_samples

    # share equally among processes & batchify
    # the code below ensures all processes have the same number of batches to generate,
    # with the same number of samples in each batch but the last one
    # (with a diff of at most 1 for the last)
    this_proc_nb_full_batches, last_batch_to_share = divmod(
        tot_nb_samples, accelerator.num_processes * eval_strat.batch_size
    )
    this_proc_last_batch, remainder = divmod(last_batch_to_share, accelerator.num_processes)
    this_proc_gen_batches = [eval_strat.batch_size] * this_proc_nb_full_batches + [this_proc_last_batch]
    if accelerator.process_index < remainder:
        this_proc_gen_batches[-1] += 1
    # pop the last batch if it is empty
    if this_proc_gen_batches[-1] == 0:
        this_proc_gen_batches.pop()

    logger.info(
        f"Process {accelerator.process_index} will generate batches of sizes: {this_proc_gen_batches} for time {true_data_class_path.name}",
        main_process_only=False,
    )
    return this_proc_gen_batches


def get_starting_batch(
    cfg: InferenceConfig,
    eval_strat: InvertedRegeneration | SimpleGeneration | ForwardNoising | InversionRegenerationOnly,
    logger: MultiProcessAdapter,
    dataset_path: Path | str,
    device: torch.device | str,
    dtype: torch.dtype | str,
):
    # build the starting dataset
    assert cfg.dataset.dataset_params is not None

    database_path = Path(cfg.dataset.path)
    logger.info(f"Using dataset {cfg.dataset.name} from {database_path}")
    subdirs: list[Path] = [e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")]
    subdirs.sort(key=cfg.dataset.dataset_params.sorting_func)
    logger.info(f"Will be using subdir '{subdirs[0].name}' as starting point if applicable")
    # ensure no transforms here
    starting_samples = list(subdirs[0].glob(f"*.{cfg.dataset.dataset_params.file_extension}"))
    kept_transforms = remove_flips_and_rotations_from_transforms(cfg.dataset.transforms)[0]
    starting_ds = cfg.dataset.dataset_params.dataset_class(
        starting_samples,
        kept_transforms,
        cfg.dataset.expected_initial_data_range,
    )
    logger.info(f"Built starting dataset from {subdirs[0]}:\n{starting_ds}")
    logger.info(f"Using transforms:\n{kept_transforms}")

    # select starting samples to generate from
    if eval_strat.plate_name_to_simulate is not None:
        # if a plate was given, select nb_generated_samples from it, in order
        glob_pattern = (
            f"{eval_strat.plate_name_to_simulate}_time_1_patch_*_*.{cfg.dataset.dataset_params.file_extension}"
        )
        # assuming they are sorted in (x,y) dictionary order
        all_matching_sample_names = list(Path(dataset_path, "1").glob(glob_pattern))
        assert len(all_matching_sample_names) > 0, f"No samples found with glob pattern {glob_pattern}"
        logger.info(f"Found {len(all_matching_sample_names)} patches from plate {eval_strat.plate_name_to_simulate}")
        # select the first nb_generated_samples
        sample_names = all_matching_sample_names[: eval_strat.nb_generated_samples]
        if len(sample_names) != len(all_matching_sample_names):
            logger.warning(f"Selected {len(sample_names)} patches out of {len(all_matching_sample_names)}")
        assert len(sample_names) == eval_strat.nb_generated_samples, (
            f"Expected to get at least {eval_strat.nb_generated_samples} samples, got {len(sample_names)}"
        )
        logger.info(f"Selected {len(sample_names)} samples to run inference from.")
        tensors: list[Tensor] = starting_ds.get_items_by_name(sample_names)
    else:
        # if not, select nb_generated_samples samples randomly
        sample_idxes: list[int] = (  # pyright: ignore[reportAssignmentType]
            np.random.default_rng().choice(len(starting_ds), eval_strat.nb_generated_samples, replace=False).tolist()
        )
        tensors = starting_ds.__getitems__(sample_idxes)
        logger.info(f"Selected {len(sample_idxes)} samples to run inference from at random")

    starting_batch: Tensor = torch.stack(tensors).to(device, dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]
    logger.debug(f"Using starting data of shape {starting_batch.shape} and type {starting_batch.dtype}")
    return starting_batch


if __name__ == "__main__":
    prof_conf = inference_conf.profiling

    logger = get_distrib_logger(inference_conf)

    if prof_conf.enabled:
        print("Profiling is enabled")

        # trace_handler = tensorboard_trace_handler("pytorch_traces", use_gzip=True) #TODO:gh-pytorch#136040
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=prof_conf.record_shapes,
            profile_memory=prof_conf.profile_memory,
            with_stack=prof_conf.with_stack,
            with_flops=prof_conf.with_flops,
            # on_trace_ready=trace_handler,
        )

        # profile the main function
        profiler.start()
        main(inference_conf, logger)
        profiler.stop()

        # torch.savetorch.full profiling trace (very large)
        if prof_conf.export_chrome_trace:
            logger.info("Saving profiling trace to profiling_trace.json...")
            profiler.export_chrome_trace("profiling_trace.json")
            logger.info("Saved profiling trace to profiling_trace.json")

        # torch.save profiling results
        logger.info("Saving top CPU calls...")
        with Path("profiling_results_self_cpu.txt").open("w") as f:
            avgs = profiler.key_averages(group_by_stack_n=5) if profiler.with_stack else profiler.key_averages()
            f.write(avgs.table(sort_by="self_cpu_time_total", row_limit=20))
        logger.info("Saved top CPU calls at profiling_results_self_cpu.txt")

        logger.info("Saving top CUDA calls...")
        with Path("profiling_results_cuda.txt").open("w") as f:
            avgs = profiler.key_averages(group_by_stack_n=5) if profiler.with_stack else profiler.key_averages()
            f.write(avgs.table(sort_by="cuda_time_total", row_limit=20))
        logger.info("Saved top CUDA calls at profiling_results_cuda.txt")

    else:
        main(inference_conf, logger)
