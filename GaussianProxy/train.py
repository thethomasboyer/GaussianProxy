# Copyright 2024 Thomas Boyer

import logging
from datetime import timedelta
from math import sqrt
from os import get_terminal_size
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.traceback import install
from termcolor import colored
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer
from wandb.sdk.wandb_run import Run as WandBRun  # pyright: ignore[reportAttributeAccessIssue]

from GaussianProxy.conf.training_conf import Config, UNet2DConditionModelConfig, UNet2DModelConfig
from GaussianProxy.utils.data import setup_dataloaders
from GaussianProxy.utils.misc import create_repo_structure, modify_args_for_debug, verify_model_configs
from GaussianProxy.utils.models import VideoTimeEncoding
from GaussianProxy.utils.training import TimeDiffusion, load_resuming_args
from my_conf.my_training_conf import config

# nice tracebacks
install()


# stop DEBUG log pollution
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("PIL").setLevel(logging.INFO)

# hardcoded config paths
DEFAULT_CONFIG_PATH = "../my_conf"
DEFAULT_CONFIG_NAME = "experiment_conf"

# Register a new resolver for torch dtype in yaml files
OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x))

# Register the config
cs = ConfigStore.instance()
cs.store(name=DEFAULT_CONFIG_NAME, node=config)


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: Config) -> None:
    # --------------------------------- Hydra Config ---------------------------------
    hydra_cfg = HydraConfig.get()

    # ------------------------------------ Logging -----------------------------------
    logger: MultiProcessAdapter = get_logger(Path(__file__).stem)

    # ---------------------------------- Accelerator ---------------------------------
    this_run_folder = Path(cfg.exp_parent_folder, cfg.project, cfg.run_name)
    accelerator_project_config = ProjectConfiguration(
        total_limit=cfg.checkpointing.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=this_run_folder.as_posix(),
    )

    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))  # 20 minutes before NCCL timeout

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with=cfg.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[accelerator_kwargs],
    )
    try:
        terminal_width = get_terminal_size().columns
    except OSError:
        terminal_width = 80
    logger.info("#" * max(0, terminal_width - 44))
    logger.info("Starting main train.py script")
    logger.info("#" * max(0, terminal_width - 44))
    logger.info(accelerator.state)

    # ----------------------------- Repository Structure -----------------------------
    (models_save_folder, saved_artifacts_folder, chckpt_save_path) = create_repo_structure(
        cfg, accelerator, logger, this_run_folder
    )
    accelerator.wait_for_everyone()

    # ---------------------------- Retrieve Resuming Args ----------------------------
    resuming_path = None  # pyright...
    if cfg.checkpointing.resume_from_checkpoint is not False:
        resuming_path, resuming_args = load_resuming_args(
            cfg,
            logger,
            accelerator,
            chckpt_save_path,
            models_save_folder,
        )
    else:
        resuming_args = None

    # ------------------------------------- WandB ------------------------------------
    # Handle run resuming with WandB
    prev_run_id = None
    prev_run_id_file = Path(accelerator_project_config.project_dir, "run_id.txt")
    if prev_run_id_file.exists():
        if cfg.checkpointing.resume_from_checkpoint is False:
            logger.warning(
                "Found a 'run_id.txt' file but 'resume_from_checkpoint' is False: ignoring this file and not resuming W&B run."
            )
        else:
            with open(prev_run_id_file, encoding="utf-8") as f:
                prev_run_id = f.readline().strip()
            logger.info(
                f"Found a 'run_id.txt' file and 'resume_from_checkpoint' is True; imposing wandb to resume the run with id {prev_run_id}"
            )
    else:
        logger.info("No 'run_id.txt' file found; starting a new W&B run")

    # Init W&B
    init_kwargs: dict[str, dict[str, str | None]] = {
        "wandb": {
            "dir": cfg.exp_parent_folder,
            "name": cfg.run_name,
            "entity": cfg.entity,
        }
    }
    if cfg.checkpointing.resume_from_checkpoint is not False and prev_run_id is not None:
        # find start step
        if resuming_args is None:
            logger.warning("No resuming state found, will rewind run from zero!")
            start_step = 0
        else:
            start_step = resuming_args.start_global_optimization_step
        # fork, rewind, or start a new run
        if cfg.accelerate.offline:
            # wandb offline run resume *simply* does not work...
            logger.warning("Offline mode: (ignoring potential previous messages and) force starting a new run")
        else:
            match cfg.resume_method:
                case "fork":
                    logger.info(f"Forking run {prev_run_id} from step {start_step}")
                    init_kwargs["wandb"]["fork_from"] = f"{prev_run_id}?_step={start_step}"
                case "rewind":
                    logger.info(f"Rewinding run {prev_run_id} from step {start_step}")
                    init_kwargs["wandb"]["resume_from"] = f"{prev_run_id}?_step={start_step}"
                case "new_run":
                    logger.info(
                        f"Starting a new run from run {prev_run_id}'s latest checkpoint, but not resuming it in wandb"
                    )
                case _:
                    raise ValueError(
                        f"Invalid resume method '{cfg.resume_method}'. Must be one of 'fork', 'rewind', or 'new_run'."
                    )

    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
        init_kwargs=init_kwargs,
    )

    # Access underlying run object on all processes (but only actually populated on main)
    wandb_tracker: WandBRun = accelerator.get_tracker("wandb", unwrap=True)  # pyright: ignore[reportAssignmentType]

    # Save the run id to a file on main process
    if accelerator.is_main_process:
        new_run_id = wandb_tracker.id
        with open(prev_run_id_file, "w", encoding="utf-8") as f:
            f.write(new_run_id)

        logger.info(
            f"Logging to: entity:{colored(cfg.entity, 'yellow', None, ['bold'])} | project:{colored(cfg.project, 'blue', None, ['bold'])} | run.name:{colored(cfg.run_name, 'magenta', None, ['bold'])} | run.id:{colored(wandb_tracker.id, 'magenta', None, ['bold'])}"
        )

    # ---------------------------------- Dataloaders ---------------------------------
    num_workers = cfg.dataloaders.num_workers if cfg.dataloaders.num_workers is not None else accelerator.num_processes

    train_dataloaders, test_dataloaders, dataset_params, fully_ordered_data = setup_dataloaders(
        cfg, accelerator, num_workers, logger, this_run_folder, chckpt_save_path, cfg.debug
    )

    # ------------------------------------ Debug -------------------------------------
    if cfg.debug:
        modify_args_for_debug(
            cfg,
            logger,
            wandb_tracker,
            accelerator.is_main_process,
            resuming_args.start_global_optimization_step if resuming_args is not None else 0,
        )

    # ------------------------------------- Net -------------------------------------
    # get net type
    if type(OmegaConf.to_object(cfg.net)) == UNet2DModelConfig:
        net_type = UNet2DModel
    elif type(OmegaConf.to_object(cfg.net)) == UNet2DConditionModelConfig:
        net_type = UNet2DConditionModel
    else:
        raise ValueError(f"Invalid type for 'cfg.net': {type(cfg.net)}")

    # load pretrained model if resuming from a "model save"
    if cfg.checkpointing.resume_from_checkpoint == "model_save" and resuming_path is not None:
        # load models
        logger.info(f"Loading pretrained net from {resuming_path / 'net'}")
        net: UNet2DModel | UNet2DConditionModel = net_type.from_pretrained(resuming_path / "net")  # pyright: ignore[reportAssignmentType]
        logger.info(f"Loading video time encoder net from {resuming_path / 'video_time_encoder'}")
        video_time_encoding: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(resuming_path / "video_time_encoder")  # pyright: ignore[reportAssignmentType]

        if accelerator.is_main_process:
            # check consistency between declared config and loaded one
            declared_net_config = net_type(**OmegaConf.to_container(cfg.net)).config  # pyright: ignore[reportCallIssue]
            verify_model_configs(net.config, declared_net_config, "net")
            declared_time_encoder_config = VideoTimeEncoding(**OmegaConf.to_container(cfg.time_encoder)).config  # pyright: ignore[reportCallIssue]
            verify_model_configs(video_time_encoding.config, declared_time_encoder_config, "time encoder")

    # else start from scratch (and maybe resume from a "proper checkpoint" just before fitting)
    else:
        # it's ugly but Hydra's instantiate produces weird errors (even with _convert="all"!?) TODO
        net = net_type(**OmegaConf.to_container(cfg.net))  # pyright: ignore[reportCallIssue]
        video_time_encoding = VideoTimeEncoding(**OmegaConf.to_container(cfg.time_encoder))  # pyright: ignore[reportCallIssue]

    nb_params_M = round(net.num_parameters() / 1e6)
    nb_params_M_trainable = round(net.num_parameters(True) / 1e6)
    logger.info(f"Net has ~{nb_params_M}M parameters (~{nb_params_M_trainable}M trainable)")

    nb_params_M = round(video_time_encoding.num_parameters() / 1e3)
    nb_params_M_trainable = round(video_time_encoding.num_parameters(True) / 1e3)
    logger.info(f"VideoTimeEncoding has ~{nb_params_M}K parameters (~{nb_params_M_trainable}K trainable)")

    # --------------------------------- Miscellaneous ---------------------------------
    # PyTorch mixed precision
    torch.set_float32_matmul_precision("high")  # replaces torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ----------------------------------- Evaluation ----------------------------------
    logger.info(f"Will use evaluation: {cfg.evaluation}")

    # ----------------------------------- Optimizer -----------------------------------
    # scale the learning rate with the square root of the number of GPUs
    logger.info(
        f"Scaling learning rate with the (square root of the) number of GPUs (×{round(sqrt(accelerator.num_processes), 3)})"
    )
    cfg.learning_rate *= sqrt(accelerator.num_processes)
    if accelerator.is_main_process:
        wandb_tracker.config.update({"learning_rate": cfg.learning_rate}, allow_val_change=True)

    optimizer: Optimizer = AdamW(
        params=list(net.parameters()) + list(video_time_encoding.parameters()),
        lr=cfg.learning_rate,
    )

    # ---------------------------- Learning Rate Scheduler ----------------------------
    lr_scheduler: LRScheduler = LinearLR(optimizer=optimizer, total_iters=cfg.training.nb_time_samplings)
    # accelerator.register_for_checkpointing(lr_scheduler)

    # ----------------------------- Distributed Compute  -----------------------------
    # We do NOT prepare *training* dataloaders! They are "fake" dataloader!

    # test dataloaders:
    for key, dl in test_dataloaders.items():
        test_dataloaders[key] = accelerator.prepare(dl)  # pyright: ignore[reportArgumentType]

    # net (includes torch.compile):
    net = accelerator.prepare(net)

    # video time encoding (includes torch.compile):
    video_time_encoding = accelerator.prepare(video_time_encoding)

    # optimizer:
    optimizer = accelerator.prepare(optimizer)

    # learning rate scheduler:
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # ------------------------------------ Dynamic -----------------------------------
    # dynamic config is never pulled from model save although it is present in the model save folder
    # => will overwrite the currently saved one (if it exists) at next model save without warning!
    dyn = DDIMScheduler(**OmegaConf.to_container(cfg.dynamic))  # pyright: ignore[reportCallIssue]

    # ------------------------------ Instantiate Trainer -----------------------------
    trainer: TimeDiffusion = TimeDiffusion(
        cfg,
        dyn,
        net,
        net_type,
        video_time_encoding,
        accelerator,
        dataset_params,
    )

    # ----------------------------- Resume Training State ----------------------------
    if (
        cfg.checkpointing.resume_from_checkpoint is not False
        and cfg.checkpointing.resume_from_checkpoint != "model_save"
        and resuming_path is not None  # pyright: ignore[reportPossiblyUnboundVariable]
    ):
        trainer.load_checkpoint(resuming_path.as_posix(), logger)  # pyright: ignore[reportPossiblyUnboundVariable]

    # ----------------------------------- Fit data  ----------------------------------
    trainer.fit(
        train_dataloaders,
        test_dataloaders,
        optimizer,
        lr_scheduler,
        logger,
        hydra_cfg.run.dir,
        models_save_folder,
        saved_artifacts_folder,
        chckpt_save_path,
        this_run_folder,
        resuming_args,
        cfg.profile,
        fully_ordered_data,
    )

    # ----------------------------------- The End  -----------------------------------
    logger.info("Ending training")
    accelerator.end_training()


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
