# Copyright 2024 Thomas Boyer

import logging
from math import sqrt
from os import get_terminal_size
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration
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
from wandb.sdk.wandb_run import Run as WandBRun

from GaussianProxy.conf.training_conf import Config, UNet2DConditionModelConfig, UNet2DModelConfig
from GaussianProxy.utils.data import setup_dataloaders
from GaussianProxy.utils.misc import args_checker, create_repo_structure, modify_args_for_debug
from GaussianProxy.utils.models import VideoTimeEncoding
from GaussianProxy.utils.training import TimeDiffusion, resume_from_checkpoint
from my_conf.my_training_conf import config

# nice tracebacks
install()

# nice wandb
wandb.require("core")

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

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with=cfg.logger,
        project_config=accelerator_project_config,
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

    # ---------------------------- Resume from Checkpoint ----------------------------
    if cfg.checkpointing.resume_from_checkpoint is not False:
        resuming_args = resume_from_checkpoint(
            cfg,
            logger,
            accelerator,
            chckpt_save_path,
        )
        logger.info(f"Loaded resuming arguments: {resuming_args}")
    else:
        resuming_args = None

    # ------------------------------------- WandB ------------------------------------
    if accelerator.is_main_process:
        # Handle run resuming
        run_id = None
        run_id_file = Path(accelerator_project_config.project_dir, "run_id.txt")
        if run_id_file.exists():
            if cfg.checkpointing.resume_from_checkpoint is False:
                logger.warning(
                    "Found a 'run_id.txt' file but 'resume_from_checkpoint' is False: ignoring this file and not resuming W&B run."
                )
            else:
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.readline().strip()
                logger.info(f"Found a 'run_id.txt' file; imposing wandb to resume the run with id {run_id}")

        # Init W&B
        init_kwargs: dict[str, dict[str, str | None]] = {
            "wandb": {
                "dir": cfg.exp_parent_folder,
                "name": cfg.run_name,
                "entity": cfg.entity,
            }
        }
        if cfg.checkpointing.resume_from_checkpoint is not False and run_id is not None:
            init_kwargs["wandb"]["id"] = run_id
            init_kwargs["wandb"]["resume"] = "must"

        accelerator.init_trackers(
            project_name=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
            # , resolve=True, throw_on_missing=True),
            # save metadata to the "wandb" directory
            # inside the *parent* folder common to all *experiments*
            init_kwargs=init_kwargs,
        )

    # Access underlying run object on all processes
    wandb_tracker: WandBRun = accelerator.get_tracker("wandb", unwrap=True)  # pyright: ignore[reportAssignmentType]

    # Save the run id to a file on main process
    if accelerator.is_main_process:
        new_run_id = wandb_tracker.id
        if run_id is not None and new_run_id != run_id:
            logger.warning(
                f"Found a 'run_id.txt' file but the run id in it ({run_id}) is different from the one generated by W&B ({new_run_id}); overwriting the file with the new run id."
            )
        with open(run_id_file, "w+", encoding="utf-8") as f:
            f.write(new_run_id)

        logger.info(
            f"Logging to: entity:{colored(cfg.entity, 'yellow', None, ['bold'])} | project:{colored(cfg.project, 'blue', None, ['bold'])} | run.name:{colored(cfg.run_name, 'magenta', None, ['bold'])} | run.id:{colored(wandb_tracker.id, 'magenta', None, ['bold'])}"
        )

    # ------------------------------------ Checks ------------------------------------
    if accelerator.is_main_process:
        args_checker(cfg, logger)

    # ---------------------------------- Dataloaders ---------------------------------
    num_workers = cfg.dataloaders.num_workers if cfg.dataloaders.num_workers is not None else accelerator.num_processes

    train_dataloaders, test_dataloaders = setup_dataloaders(cfg, num_workers, logger, cfg.debug)

    # ------------------------------------ Debug -------------------------------------
    if cfg.debug:
        modify_args_for_debug(cfg, logger, wandb_tracker, accelerator.is_main_process)
        if accelerator.is_main_process:
            args_checker(cfg, logger, False)  # check again after debug modifications 😈

    # ------------------------------------- Net -------------------------------------
    # it's ugly but Hydra's instantiate produces weird errors (even with _convert="all"!?) TODO
    # also need saving model here type because that info is apparently lost when compiling?!
    if type(OmegaConf.to_object(cfg.net)) == UNet2DModelConfig:
        net = UNet2DModel(**OmegaConf.to_container(cfg.net))  # pyright: ignore[reportCallIssue]
        net_type = UNet2DModel
    elif type(OmegaConf.to_object(cfg.net)) == UNet2DConditionModelConfig:
        net = UNet2DConditionModel(**OmegaConf.to_container(cfg.net))  # pyright: ignore[reportCallIssue]
        net_type = UNet2DConditionModel
    else:
        raise ValueError(f"Invalid type for 'cfg.net': {type(cfg.net)}")

    # video time encoding
    video_time_encoding = VideoTimeEncoding(**OmegaConf.to_container(cfg.time_encoder))  # pyright: ignore[reportCallIssue]

    # --------------------------------- Miscellaneous --------------------------------
    # # Create EMA for the models
    # ema_models = {}
    # components_to_train_transcribed = get_HF_component_names(cfg.components_to_train)
    # if cfg.use_ema:
    #     for module_name, module in pipeline.components.items():
    #         if module_name in components_to_train_transcribed:
    #             ema_models[module_name] = EMAModel(
    #                 module.parameters(),
    #                 decay=cfg.ema_max_decay,
    #                 use_ema_warmup=True,
    #                 inv_gamma=cfg.ema_inv_gamma,
    #                 power=cfg.ema_power,
    #                 model_cls=module.__class__,
    #                 model_config=module.config,
    #             )
    #             ema_models[module_name].to(accelerator.device)
    #     logger.info(
    #         f"Created EMA weights for the following models: {list(ema_models)} (corresponding to the (unordered) following cfg: {cfg.components_to_train})"
    #     )
    # Log models gradients
    if accelerator.is_main_process:
        wandb.watch(accelerator.unwrap_model(net), "gradients", log_freq=1000, idx=0)

    # PyTorch mixed precision
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # ----------------------------------- Optimizers ----------------------------------
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
    # We do NOT prepare *training* dataloaders!
    # they are "fake" dataloader!

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

    # ----------------------------- Initial best metrics -----------------------------
    # if accelerator.is_main_process:
    #     best_metric = get_initial_best_metric()

    # ------------------------------------ Dynamic -----------------------------------
    dyn = DDIMScheduler(**OmegaConf.to_container(cfg.dynamic))  # pyright: ignore[reportCallIssue]

    # --------------------------------- Training loop --------------------------------
    trainer: TimeDiffusion = TimeDiffusion(
        dynamic=dyn,
        net=net,
        net_type=net_type,
        video_time_encoding=video_time_encoding,
        accelerator=accelerator,
        debug=cfg.debug,
    )
    trainer.fit(
        train_dataloaders,
        test_dataloaders,
        optimizer,
        lr_scheduler,
        logger,
        hydra_cfg.run.dir,
        models_save_folder,
        saved_artifacts_folder,
        cfg.training,
        cfg.checkpointing,
        chckpt_save_path,
        cfg.evaluation,
        resuming_args,
        cfg.profile,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
