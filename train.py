# Copyright 2024 Thomas Boyer

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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.traceback import install
from termcolor import colored
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from wandb.sdk.wandb_run import Run as WandBRun

from conf.conf import Config
from my_conf.experiment_conf import config
from utils.data import setup_dataloaders
from utils.misc import args_checker, create_repo_structure, modify_args_for_debug
from utils.models import VideoTimeEncoding
from utils.training import TimeDiffusion, resume_from_checkpoint

# nice tracebacks
install()

# hardcoded config paths
DEFAULT_CONFIG_PATH = "my_conf"
DEFAULT_CONFIG_NAME = "experiment_conf"

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
    (
        models_save_folder,
        saved_artifacts_folder,
    ) = create_repo_structure(cfg, accelerator, logger, this_run_folder)
    accelerator.wait_for_everyone()

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
            elif cfg.checkpointing.resume_from_checkpoint is True:
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.readline().strip()
                logger.info(f"Found a 'run_id.txt' file; imposing wandb to resume the run with id {run_id}")
            else:
                raise ValueError(
                    f"Invalid value for 'resume_from_checkpoint' argument: {cfg.checkpointing.resume_from_checkpoint}"
                )

        # Init W&B
        init_kwargs: dict[str, dict[str, str | None]] = {
            "wandb": {
                "dir": cfg.exp_parent_folder,
                "name": cfg.run_name,
                "entity": cfg.entity,
            }
        }
        if cfg.checkpointing.resume_from_checkpoint is True and run_id is not None:
            init_kwargs["wandb"]["id"] = run_id
            init_kwargs["wandb"]["resume"] = "must"

        accelerator.init_trackers(
            project_name=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            # , resolve=True, throw_on_missing=True),
            # save metadata to the "wandb" directory
            # inside the *parent* folder common to all *experiments*
            init_kwargs=init_kwargs,
        )

    # Access underlying run object on all processes
    wandb_tracker: WandBRun = accelerator.get_tracker("wandb", unwrap=True)  # type: ignore

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

    train_datasets, test_dataloaders = setup_dataloaders(cfg, num_workers, logger, cfg.debug)

    # ------------------------------------ Debug -------------------------------------
    if cfg.debug:
        modify_args_for_debug(cfg, logger, wandb_tracker, accelerator.is_main_process)
        if accelerator.is_main_process:
            args_checker(cfg, logger, False)  # check again after debug modifications 😈

    # ------------------------------------- Net -------------------------------------
    net = UNet2DModel(**OmegaConf.to_container(cfg.net))  # type: ignore

    # video time encoding
    num_channels_out_of_first_block_UNet = net.time_proj.num_channels
    video_time_encoding = VideoTimeEncoding(
        encoding_dim=num_channels_out_of_first_block_UNet,
        time_embed_dim=num_channels_out_of_first_block_UNet * 4,
        flip_sin_to_cos=True,
        downscale_freq_shift=1,
    )

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
        params=net.parameters(),
        lr=cfg.learning_rate,
        fused=cfg.adam_use_fused,
    )

    # ---------------------------- Learning Rate Scheduler ----------------------------
    lr_scheduler: LRScheduler = LinearLR(optimizer=optimizer, total_iters=cfg.training.nb_time_samplings)
    # accelerator.register_for_checkpointing(lr_scheduler)

    # ----------------------------- Distributed Compute  -----------------------------
    # dataloaders (test only, train is raw dataset):
    for key, dl in test_dataloaders.items():
        test_dataloaders[key] = accelerator.prepare(dl)

    # net (includes torch.compile):
    net = accelerator.prepare(net)

    # video time encoding (includes torch.compile):
    video_time_encoding = accelerator.prepare(video_time_encoding)

    # optimizer:
    optimizer = accelerator.prepare(optimizer)

    # learning rate scheduler:
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # ---------------------------- Resume from Checkpoint ----------------------------
    if cfg.checkpointing.resume_from_checkpoint is not False:
        resuming_args = resume_from_checkpoint(
            cfg,
            logger,
            accelerator,
        )
        logger.info(f"Loaded resuming arguments: {resuming_args}")
    else:
        resuming_args = None

    # ----------------------------- Initial best metrics -----------------------------
    # if accelerator.is_main_process:
    #     best_metric = get_initial_best_metric()

    # ------------------------------------ Dynamic -----------------------------------
    dyn = DDIMScheduler(**vars(cfg.dynamic))

    # --------------------------------- Training loop --------------------------------
    trainer: TimeDiffusion = TimeDiffusion(
        dynamic=dyn,
        net=net,
        video_time_encoding=video_time_encoding,
        accelerator=accelerator,
        debug=cfg.debug,
    )
    trainer.fit(
        train_datasets,
        test_dataloaders,
        optimizer,
        lr_scheduler,
        logger,
        hydra_cfg.run.dir,
        models_save_folder,
        saved_artifacts_folder,
        cfg.training,
        cfg.checkpointing,
        resuming_args,
        cfg.profile,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
