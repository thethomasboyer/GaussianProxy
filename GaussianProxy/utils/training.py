import json
import pickle
import shutil
from dataclasses import dataclass, field, fields

# from logging import INFO, FileHandler, makeLogRecord
from pathlib import Path
from typing import Generator, Optional, Type

import numpy as np
import torch
import torch_fidelity
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from enlighten import Manager, get_manager
from torch import IntTensor, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.profiler import profile as torch_profile
from torch.profiler import schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

from GaussianProxy.conf.training_conf import (
    Config,
    FIDComputation,
    ForwardNoising,
    InvertedRegeneration,
    IterativeInvertedRegeneration,
    SimpleGeneration,
)
from GaussianProxy.utils.data import BaseDataset
from GaussianProxy.utils.misc import (
    StateLogger,
    get_evenly_spaced_timesteps,
    log_state,
    save_eval_artifacts_log_to_wandb,
    save_images_for_fid_compute,
)
from GaussianProxy.utils.models import VideoTimeEncoding

# State logger to track time spent in some functions
state_logger = StateLogger()


@dataclass
class ResumingArgs:
    """
    Arguments to resume training from a checkpoint.

    Must default to "natural" starting values when training from scratch.
    """

    start_instant_batch_idx: int = 0
    start_global_optimization_step: int = 0
    best_model_to_date: bool = True


@dataclass
class TimeDiffusion:
    """
    A diffusion model with an additional time conditioning for per-sample video generation

    What's used for the encoding of time passed to the model is a Transformer-like sinusoidal encoding followed by a simple MLP,
    which is actually exactly the same architecture as for the diffusion discretization timesteps.

    Note that "time" here refers to the "biological process" time, or "video" time,
    *not* the "diffusion process" discretization time(step).
    """

    # compulsory arguments
    cfg: Config
    dynamic: DDIMScheduler
    net: UNet2DModel | UNet2DConditionModel
    net_type: Type[UNet2DModel | UNet2DConditionModel]
    video_time_encoding: VideoTimeEncoding
    accelerator: Accelerator
    debug: bool
    # populated arguments when calling .fit
    chckpt_save_path: Path = field(init=False)
    optimizer: Optimizer = field(init=False)
    lr_scheduler: LRScheduler = field(init=False)
    logger: MultiProcessAdapter = field(init=False)
    output_dir: str = field(init=False)
    model_save_folder: Path = field(init=False)
    saved_artifacts_folder: Path = field(init=False)
    # inferred constant attributes
    _nb_empirical_dists: int = field(init=False)
    _empirical_dists_timesteps: list[float] = field(init=False)
    _data_shape: tuple[int, int, int] = field(init=False)
    # resuming args
    _resuming_args: ResumingArgs = field(init=False)
    # instant state for checkpointing
    instant_batch_idx: int = field(init=False)
    # global training state
    global_optimization_step: int = field(init=False)
    best_model_to_date: bool = True  # TODO
    # constant evaluation starting states
    _eval_noise: Tensor = field(init=False)
    _eval_video_times: Tensor = field(init=False)

    @property
    def nb_empirical_dists(self) -> int:
        if not hasattr(self, "_nb_empirical_dists"):
            raise RuntimeError("nb_empirical_dists is not set; fit should be called before trying to access it")
        return self._nb_empirical_dists

    @property
    def empirical_dists_timesteps(self) -> list[float]:
        if self._empirical_dists_timesteps is None:
            raise RuntimeError("empirical_dists_timesteps is not set; fit should be called before trying to access it")
        return self._empirical_dists_timesteps

    @property
    def resuming_args(self) -> ResumingArgs:
        if self._resuming_args is None:
            raise RuntimeError("resuming_args is not set; fit should be called before trying to access it")
        return self._resuming_args

    @property
    def data_shape(self) -> tuple[int, int, int]:
        if self._data_shape is None:
            raise RuntimeError("data_shape is not set; fit should be called before trying to access it")
        return self._data_shape

    def get_eval_noise(self) -> Tensor:
        if self._eval_noise is None:
            raise RuntimeError("eval_noise is not set; fit should be called before trying to access it")
        return self._eval_noise.clone()

    def get_eval_video_times(self) -> Tensor:
        if self._eval_video_times is None:
            raise RuntimeError("eval_video_times is not set; fit should be called before trying to access it")
        return self._eval_video_times.clone()

    def __post_init__(self):
        # Assign the appropriate function based on net_type
        if self.net_type is UNet2DModel:
            self._net_pred = self._unet2d_pred
        elif self.net_type is UNet2DConditionModel:
            self._net_pred = self._unet2d_condition_pred
        else:
            raise ValueError(f"Expecting UNet2DModel or UNet2DConditionModel, got {self.net_type}")

    def fit(
        self,
        train_dataloaders: dict[int, DataLoader] | dict[str, DataLoader],
        test_dataloaders: dict[int, DataLoader] | dict[str, DataLoader],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: MultiProcessAdapter,
        output_dir: str,
        model_save_folder: Path,
        saved_artifacts_folder: Path,
        chckpt_save_path: Path,
        resuming_args: Optional[ResumingArgs] = None,
        profile: bool = False,
    ):
        """
        Global high-level fitting method.
        """
        logger.debug(
            f"Starting TimeDiffusion fitting on process {self.accelerator.process_index}",
            main_process_only=False,
        )
        # Set some attributes relative to the data
        self._fit_init(
            train_dataloaders,
            chckpt_save_path,
            optimizer,
            lr_scheduler,
            logger,
            model_save_folder,
            saved_artifacts_folder,
            resuming_args,
        )

        # Modify dataloaders dicts to use the empirical distribution timesteps as keys, instead of a mere numbering
        train_timestep_dataloaders: dict[float, DataLoader] = {}
        test_timestep_dataloaders: dict[float, DataLoader] = {}
        for split, dls in [("train", train_dataloaders), ("test", test_dataloaders)]:
            self.logger.info(f"Using {split} dataloaders ordering: {list(dls.keys())}")
            timestep_dataloaders = train_timestep_dataloaders if split == "train" else test_timestep_dataloaders
            for dataloader_idx, dl in enumerate(dls.values()):
                timestep_dataloaders[self.empirical_dists_timesteps[dataloader_idx]] = dl
            assert (
                len(timestep_dataloaders) == self.nb_empirical_dists == len(dls)
            ), f"Got {len(timestep_dataloaders)} dataloaders, nb_empirical_dists={self.nb_empirical_dists} and len(dls)={len(dls)}; they should be equal"
            assert np.all(
                np.diff(list(timestep_dataloaders.keys())) > 0
            ), "Expecting newly key-ed dataloaders to be numbered in increasing order."

        pbar_manager: Manager = get_manager()  # pyright: ignore[reportAssignmentType]

        # profiling # TODO: move to train.py
        if profile:
            profiler = torch_profile(
                schedule=schedule(skip_first=1, wait=1, warmup=3, active=20, repeat=3),
                on_trace_ready=tensorboard_trace_handler(Path(output_dir, "profile").as_posix()),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()
        else:
            profiler = None

        init_count = self.resuming_args.start_instant_batch_idx
        batches_pbar = pbar_manager.counter(
            total=self.cfg.training.nb_time_samplings,
            position=1,
            desc="Training batches" + 10 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
            count=init_count,
        )
        if self.accelerator.is_main_process:
            batches_pbar.refresh()
        # loop through training batches
        for batch_idx, (time, batch) in enumerate(
            self._yield_data_batches(
                train_timestep_dataloaders,
                self.logger,
                self.cfg.training.nb_time_samplings - init_count,
            )
        ):
            self.instant_batch_idx = batch_idx
            # set network to the right mode
            self.net.train()
            # gradient step here
            self._fit_one_batch(batch, time)
            # take one profiler step
            if profiler is not None:
                profiler.step()
            # checkpoint
            if self.global_optimization_step % self.cfg.checkpointing.checkpoint_every_n_steps == 0:
                self._checkpoint()
            # update pbar
            batches_pbar.update()
            # evaluate models every n optimisation steps
            if (
                self.cfg.evaluation.every_n_opt_steps is not None
                and self.global_optimization_step % self.cfg.evaluation.every_n_opt_steps == 0
            ):
                self._evaluate(
                    test_timestep_dataloaders,
                    pbar_manager,
                )
        batches_pbar.close()
        # update timeline & save it # TODO: broken since epochs removal; to update every n steps (and to fix...)
        gantt_chart = state_logger.create_gantt_chart()
        self.accelerator.log({"state_timeline/": wandb.Plotly(gantt_chart)}, step=self.global_optimization_step)

        if profiler is not None:
            profiler.stop()

    def _fit_init(
        self,
        train_dataloaders: dict[int, DataLoader] | dict[str, DataLoader],
        chckpt_save_path: Path,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: MultiProcessAdapter,
        model_save_folder: Path,
        saved_artifacts_folder: Path,
        resuming_args: Optional[ResumingArgs],
    ):
        """Fit some remaining attributes before fitting."""
        assert not hasattr(self, "._nb_empirical_dists"), "Already fitted"
        self._nb_empirical_dists = len(train_dataloaders)
        assert self._nb_empirical_dists > 1, "Expecting at least 2 empirical distributions to train the model."
        self._empirical_dists_timesteps = get_evenly_spaced_timesteps(self.nb_empirical_dists)
        self.chckpt_save_path = chckpt_save_path
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.model_save_folder = model_save_folder
        self.saved_artifacts_folder = saved_artifacts_folder
        # resuming args
        if resuming_args is None:
            self._resuming_args = ResumingArgs()
        else:
            self._resuming_args = resuming_args
        self.global_optimization_step = self.resuming_args.start_global_optimization_step
        self.best_model_to_date = self.resuming_args.best_model_to_date
        # expected data shape
        unwrapped_net_config = self.accelerator.unwrap_model(self.net).config
        self._data_shape = (
            unwrapped_net_config["in_channels"],
            unwrapped_net_config["sample_size"],
            unwrapped_net_config["sample_size"],
        )
        # Sample Gaussian noise and video timesteps once for all subsequent evaluations
        self._eval_noise = torch.randn(
            self.cfg.evaluation.batch_size,
            self.accelerator.unwrap_model(self.net).config["in_channels"],
            self.accelerator.unwrap_model(self.net).config["sample_size"],
            self.accelerator.unwrap_model(self.net).config["sample_size"],
            device=self.accelerator.device,
        )
        eval_video_times = torch.rand(
            self.cfg.evaluation.batch_size,
            device=self.accelerator.device,
        )
        self._eval_video_times = torch.sort(eval_video_times).values  # sort it for better viz

    @torch.no_grad()
    @log_state(state_logger)
    def _yield_data_batches(
        self,
        dataloaders: dict[float, DataLoader],
        logger: MultiProcessAdapter,
        nb_time_samplings: int,
    ) -> Generator[tuple[float, Tensor], None, None]:
        """
        Yield `nb_time_samplings` data batches from the given dataloaders.

        Each batch has an associated timestep uniformly sampled between 0 and 1,
        and the corresponding batch is formed by "interpolating" the two empirical distributions
        of closest timesteps, where "interpolating" means that for timestep:

        t = x * t- + (1-x) * t+

        where x ∈ [0;1], and t- and t+ are the two immediately inferior and superior empirical timesteps to t, the
        returned batch is formed by sampling true data samples of empirical distributions t- and t+
        with probability x and 1-x, respectively.

        Note that such behavior is only "stable" when the batch size is large enough. Typically, if batch size is 1
        then each batch will always have samples from one empirical distribution only: quite bad for training along
        continuous time... Along many samplings the theoretical end result will of course be the same.
        """
        assert (
            list(dataloaders.keys()) == self.empirical_dists_timesteps
            and list(dataloaders.keys()) == sorted(dataloaders.keys())
        ), f"Expecting dataloaders to be ordered by timestep, got list(dataloaders.keys())={list(dataloaders.keys())} vs self.empirical_dists_timesteps={self.empirical_dists_timesteps}"

        dataloaders_iterators = {t: iter(dl) for t, dl in dataloaders.items()}

        for _ in range(nb_time_samplings):
            # sample a time between 0 and 1
            t = torch.rand(1).item()

            # get the two closest empirical distributions
            t_minus, t_plus = None, None
            for i in range(len(self.empirical_dists_timesteps) - 1):
                if self.empirical_dists_timesteps[i] <= t < self.empirical_dists_timesteps[i + 1]:
                    t_minus = self.empirical_dists_timesteps[i]
                    t_plus = self.empirical_dists_timesteps[i + 1]
                    break
            assert (
                t_minus is not None and t_plus is not None
            ), f"Could not find the two closest empirical distributions for time {t}"

            # get distance from each time (x such that t = x * t- + (1-x) * t+)
            x = (t_plus - t) / (t_plus - t_minus)

            # form the batch
            batch = []
            # sample from t- with probability x, from t+ with probability 1-x
            random_sampling = torch.rand((self.cfg.training.train_batch_size,))

            # now sample from the two dataloaders
            nb_t_minus_samples = (random_sampling < x).int().sum().item()
            nb_t_plus_samples = self.cfg.training.train_batch_size - nb_t_minus_samples
            nb_samples_to_get = {t_minus: nb_t_minus_samples, t_plus: nb_t_plus_samples}
            nb_samples_got = {t_minus: 0, t_plus: 0}

            for t_to_sample_from in (t_minus, t_plus):
                dl_to_sample_from = dataloaders_iterators[t_to_sample_from]
                while nb_samples_got[t_to_sample_from] != nb_samples_to_get[t_to_sample_from]:
                    # Sample a new batch and add it to the batch
                    try:
                        max_nb_samples_to_get = nb_samples_to_get[t_to_sample_from] - nb_samples_got[t_to_sample_from]
                        new_sample = next(dl_to_sample_from)[:max_nb_samples_to_get]
                        batch += new_sample
                        nb_samples_got[t_to_sample_from] += len(new_sample)
                    # DataLoaders will quickly be exhausted (resulting in silent hangs then undebuggable NCCL timeouts 🙃)
                    # so reform them when needed
                    except StopIteration:
                        self.logger.debug(
                            f"Reforming dataloader for timestep {t_to_sample_from} on process {self.accelerator.process_index}",
                            main_process_only=False,
                        )
                        dataloaders_iterators[t_to_sample_from] = iter(dataloaders[t_to_sample_from])
                        dl_to_sample_from = dataloaders_iterators[t_to_sample_from]

            # now concatenate tensors and manually move to device
            # since training dataloaders are not prepared
            batch = torch.stack(batch).to(self.accelerator.device)

            # finally shuffle the batch so that seen *empirical times* are mixed
            # (otherwise we'd always have samples from t- before those from t+,
            # not that bad but it's probably better to interleave as much as possible)
            batch = batch[torch.randperm(batch.shape[0])]

            # check shape and append
            assert batch.shape == (
                self.cfg.training.train_batch_size,
                *self.data_shape,
            ), f"Expecting sample shape {(self.cfg.training.train_batch_size, *self.data_shape)}, got {batch.shape}"

            yield t, batch

    @log_state(state_logger)
    def _fit_one_batch(
        self,
        batch: Tensor,
        time: float,
    ):
        """
        Perform one training step on one batch at one timestep.

        Computes the loss w.r.t. true targets and backpropagates the error
        """
        # checks
        assert batch.shape == (
            self.cfg.training.train_batch_size,
            *self.data_shape,
        ), f"Expecting batch shape {(self.cfg.training.train_batch_size, *self.data_shape)}, got {batch.shape}"
        assert (
            batch.min() >= -1 and batch.max() <= 1
        ), f"Expecting batch to be in [-1;1] range, got {batch.min()} and {batch.max()}"

        # Sample Gaussian noise
        noise = torch.randn_like(batch)

        # Sample a random diffusion timestep
        diff_timesteps: IntTensor = torch.randint(  # pyright: ignore[reportAssignmentType]
            0,
            self.dynamic.config["num_train_timesteps"],
            (batch.shape[0],),
            device=self.accelerator.device,
        )

        # Forward diffusion process
        noisy_batch = self.dynamic.add_noise(batch, noise, diff_timesteps)  # pyright: ignore[reportArgumentType]

        # Encode time
        video_time_codes = self.video_time_encoding.forward(time, batch.shape[0])

        # Get model predictions
        pred = self._net_pred(noisy_batch, diff_timesteps, video_time_codes)

        # Get target
        match self.dynamic.config["prediction_type"]:
            case "epsilon":
                target = noise
            case "v_prediction":
                target = self.dynamic.get_velocity(batch, noise, diff_timesteps)
            case _:
                raise ValueError(
                    f"Expected self.dynamic.config.prediction_type to be 'epsilon' or 'v_prediction', got '{self.dynamic.config['prediction_type']}'"
                )

        # Compute loss
        loss = self._loss(pred, target)

        # Wake me up at 3am if loss is NaN
        if torch.isnan(loss) and self.accelerator.is_main_process:
            msg = f"loss is NaN at step {self.global_optimization_step}, time {time}"
            wandb.alert(  # pyright: ignore[reportAttributeAccessIssue]
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,  # pyright: ignore[reportAttributeAccessIssue]
                wait_duration=21600,  # 6 hours
            )
            self.logger.critical(msg)
            # TODO: restart from previous checkpoint?

        # Backward pass
        self.accelerator.backward(loss)

        # Gradient clipping
        grad_norm = None  # pyright: ignore[reportAssignmentType]
        if self.accelerator.sync_gradients:
            grad_norm: Tensor = self.accelerator.clip_grad_norm_(  # pyright: ignore[reportAssignmentType]
                self.net.parameters(),
                self.cfg.training.max_grad_norm,
            )
            # Wake me up at 3am if grad is NaN
            if torch.isnan(grad_norm) and self.accelerator.is_main_process:
                msg = f"Grad is NaN at step {self.global_optimization_step}, time {time}"
                if self.accelerator.scaler is not None:
                    msg += f", grad scaler scale {self.accelerator.scaler.get_scale()} from initial scale {self.accelerator.scaler._init_scale}"
                wandb.alert(  # pyright: ignore[reportAttributeAccessIssue]
                    title="NaN grad",
                    text=msg,
                    level=wandb.AlertLevel.ERROR,  # pyright: ignore[reportAttributeAccessIssue]
                    wait_duration=21600,  # 6 hours
                )
                self.logger.critical(msg)
                # TODO: restart from previous checkpoint?

        # Optimization step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Log to wandb
        self.accelerator.log(
            {
                "training/loss": loss.item(),
                "training/lr": self.lr_scheduler.get_last_lr()[0],
                "training/step": self.global_optimization_step,
                "training/time": time,
                "training/L2 gradient norm": grad_norm,
            },
            step=self.global_optimization_step,
        )
        # Wake me up at 3am if loss is NaN
        if torch.isnan(loss) and self.accelerator.is_main_process:
            msg = f"Loss is NaN at step {self.global_optimization_step}, time {time}"
            wandb.alert(  # pyright: ignore[reportAttributeAccessIssue]
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,  # pyright: ignore[reportAttributeAccessIssue]
                wait_duration=21600,  # 6 hours
            )
            self.logger.critical(msg)
            # TODO: restart from previous checkpoint

        # Update global opt step
        self.global_optimization_step += 1

    def _unet2d_pred(
        self,
        noisy_batch: Tensor,
        diff_timesteps: Tensor,
        video_time_codes: Tensor,
        net: Optional[UNet2DModel] = None,
    ):
        # allow providing evaluation net
        _net: UNet2DModel = self.net if net is None else net  # pyright: ignore[reportAssignmentType]
        return _net.forward(
            noisy_batch,
            diff_timesteps,
            class_labels=video_time_codes,
            return_dict=False,
        )[0]

    def _unet2d_condition_pred(
        self,
        noisy_batch: Tensor,
        diff_timesteps: Tensor,
        video_time_codes: Tensor,
        net: Optional[UNet2DConditionModel] = None,
    ):
        # allow providing evaluation net
        _net: UNet2DConditionModel = self.net if net is None else net  # pyright: ignore[reportAssignmentType]
        return _net.forward(
            noisy_batch,
            diff_timesteps,
            encoder_hidden_states=video_time_codes.unsqueeze(1),
            return_dict=False,
        )[0]

    def _loss(self, pred, target):
        """All the hard work should happen before..."""
        criterion = torch.nn.MSELoss()
        loss = criterion(pred, target)
        return loss

    @torch.inference_mode()
    def _evaluate(
        self,
        dataloaders: dict[float, DataLoader[BaseDataset]],
        pbar_manager: Manager,
    ):
        """
        Generate inference trajectories, compute metrics and save the model if best to date.

        Should be called by all processes as distributed barriers are used here.
        Distributed processing is handled herein.
        """
        # 0. Wait for all processes to reach this point
        self.accelerator.wait_for_everyone()

        # 1. Save instantenous states of models
        # tmp save folder in the checkpoints folder
        tmp_save_folder = self.chckpt_save_path / ".tmp_inference_save"
        if self.accelerator.is_main_process:
            if tmp_save_folder.exists():
                shutil.rmtree(tmp_save_folder)
            else:
                tmp_save_folder.mkdir()
        self.accelerator.wait_for_everyone()
        # save net & video time encoder
        self.accelerator.unwrap_model(self.net).save_pretrained(
            tmp_save_folder / "net", is_main_process=self.accelerator.is_main_process
        )
        self.accelerator.unwrap_model(self.video_time_encoding).save_pretrained(
            tmp_save_folder / "video_time_encoder",
            is_main_process=self.accelerator.is_main_process,
        )
        self.accelerator.wait_for_everyone()

        # 2. Instantiate inference models
        if self.net_type == UNet2DModel:
            inference_net: UNet2DModel = UNet2DModel.from_pretrained(  # pyright: ignore[reportAssignmentType, reportRedeclaration]
                tmp_save_folder / "net",
                local_files_only=True,
                device_map=self.accelerator.process_index,
            )
        elif self.net_type == UNet2DConditionModel:
            inference_net: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(  # pyright: ignore[reportAssignmentType]
                tmp_save_folder / "net",
                local_files_only=True,
                device_map=self.accelerator.process_index,
            )
        else:
            raise ValueError(f"Expecting UNet2DModel or UNet2DConditionModel, got {self.net_type}")
        inference_video_time_encoding: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(  # pyright: ignore[reportAssignmentType]
            tmp_save_folder / "video_time_encoder",
            local_files_only=True,
            device_map=self.accelerator.process_index,
        )
        # TODO: save & load a compiled artifact (with torch.export?)

        # 3. Run through evaluation strategies
        for eval_strat in self.cfg.evaluation.strategies:
            if eval_strat.name == "SimpleGeneration":
                self._simple_gen(
                    dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )
            elif eval_strat.name == "ForwardNoising":
                self._forward_noising(
                    dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )
            elif eval_strat.name == "InvertedRegeneration":
                self._inv_regen(
                    dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )
            elif eval_strat.name == "IterativeInvertedRegeneration":
                self._iter_inv_regen(
                    dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )
            elif eval_strat.name == "FIDComputation":
                true_data_classes_paths = {idx: "" for idx in range(len(dataloaders.keys()))}
                for cl_idx, dl in enumerate(dataloaders.values()):
                    inner_ds = dl.dataset
                    if not isinstance(inner_ds, BaseDataset):
                        raise ValueError(
                            f"Expected a `BaseDataset` for the underlying dataset of the evaluation dataloader, got {type(inner_ds)}"
                        )
                    # assuming they are share the same parent...
                    true_data_classes_paths[cl_idx] = Path(inner_ds.samples[0]).parent.as_posix()
                self._fid_computation(
                    tmp_save_folder,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                    true_data_classes_paths,
                )
            else:
                raise ValueError(f"Unknown evaluation strategy {eval_strat}")

            # wait for everyone between each eval
            self.accelerator.wait_for_everyone()

        # Save best model
        if self.best_model_to_date:
            self._save_pipeline()

    @log_state(state_logger)
    @torch.inference_mode()
    def _simple_gen(
        self,
        _: dict[float, DataLoader],
        pbar_manager: Manager,
        eval_strat: SimpleGeneration,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        Just simple generations.
        """
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

        # generate time encodings
        random_video_time = self.get_eval_video_times()
        random_video_time_enc = inference_video_time_encoding.forward(random_video_time)

        # generate a sample
        gen_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Generating samples" + " " * 8,
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        if self.accelerator.is_main_process:
            gen_pbar.refresh()

        image = self.get_eval_noise()
        for t in inference_scheduler.timesteps:
            model_output = self._net_pred(image, t, random_video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            gen_pbar.update()

        save_eval_artifacts_log_to_wandb(
            image,
            self.saved_artifacts_folder,
            self.global_optimization_step,
            self.accelerator,
            self.logger,
            eval_strat.name,
            "simple_generations",
            ["-1_1 raw", "image min-max", "-1_1 clipped"],
            captions=[f"time: {round(t.item(), 3)}" for t in random_video_time],
        )

        gen_pbar.close()

    @log_state(state_logger)
    @torch.inference_mode()
    def _cosine_sim_with_train(
        self,
        _: dict[float, DataLoader],
        pbar_manager: Manager,
        eval_strat: SimpleGeneration,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        Test model memorization by:

        - generating n samples (from random noise)
        - computing the closest (generated, true) pair by Euclidean cosine similarity, for each generated image
        - plotting the distribution of these n closest cosine similarities
        - showing the p < n closest pairs
        """
        raise NotImplementedError("TODO: Not implemented yet")

    @log_state(state_logger)
    @torch.inference_mode()
    def _inv_regen(
        self,
        dataloaders: dict[float, DataLoader],
        pbar_manager: Manager,
        eval_strat: InvertedRegeneration,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        Two steps are performed on each batch of the first dataloader:
            1. Perform inversion to obtain the starting Gaussian
            2. Generate te trajectory from that inverted Gaussian sample

        Note that any other dataloader than the first one is actually not used for evaluation.
        """
        # Checks
        assert (
            list(dataloaders.keys())[0] == 0
        ), f"Expecting the first dataloader to be at time 0, got {list(dataloaders.keys())[0]}"

        # Setup schedulers
        inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

        # Use only 1st dataloader for now TODO
        first_dl = list(dataloaders.values())[0]
        eval_batches_pbar = pbar_manager.counter(
            total=len(first_dl),
            position=2,
            desc="Evaluation batches" + 10 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        if self.accelerator.is_main_process:
            eval_batches_pbar.refresh()

        # Now generate & evaluate the trajectories
        for batch_idx, batch in enumerate(iter(first_dl)):
            # 1. Save the to-be inverted images
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    batch,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "starting_samples",
                    ["-1_1 raw", "image min-max"],
                )

            # 2. Generate the inverted Gaussians
            inverted_gauss = batch
            inversion_video_time = inference_video_time_encoding.forward(0, batch.shape[0])

            for t in inverted_scheduler.timesteps:
                model_output = self._net_pred(inverted_gauss, t, inversion_video_time, eval_net)  # pyright: ignore[reportArgumentType]
                inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]

            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    inverted_gauss,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "inversions",
                    ["image min-max", "image 5perc-95perc"],
                )

            # 3. Regenerate the starting samples from their inversion
            regen = inverted_gauss.clone()
            for t in inference_scheduler.timesteps:
                model_output = self._net_pred(regen, t, inversion_video_time, eval_net)  # pyright: ignore[reportArgumentType]
                regen = inference_scheduler.step(model_output, int(t), regen, return_dict=False)[0]
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    regen,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "regenerations",
                    ["image min-max", "-1_1 raw", "-1_1 clipped"],
                )

            # 4. Generate the trajectory from it
            # TODO: parallelize the generation along video time?
            # usefull if small inference batch size, otherwise useless
            video = []
            video_time_pbar = pbar_manager.counter(
                total=self.cfg.evaluation.nb_video_timesteps,
                position=3,
                desc="Evaluation video timesteps",
                enable=self.accelerator.is_main_process,
                leave=False,
                min_delta=1,
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            for video_t_idx, video_time in enumerate(torch.linspace(0, 1, self.cfg.evaluation.nb_video_timesteps)):
                image = inverted_gauss.clone()
                self.logger.debug(f"Video timestep index {video_t_idx} / {self.cfg.evaluation.nb_video_timesteps - 1}")
                video_time_enc = inference_video_time_encoding.forward(video_time.item(), batch.shape[0])

                for t in inference_scheduler.timesteps:
                    model_output = self._net_pred(image, t, video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                video.append(image)
                video_time_pbar.update()
                # _log_to_file_only(
                #     self.logger,
                #     f"Video time {video_timestep}/{self.cfg.training.eval_nb_video_timesteps}",
                #     INFO,
                # )

            video_time_pbar.close(clear=True)

            video = torch.stack(video)
            # wandb expects (batch_size, video_time, channels, height, width)
            video = video.permute(1, 0, 2, 3, 4)
            expected_video_shape = (
                self.cfg.evaluation.batch_size,
                self.cfg.evaluation.nb_video_timesteps,
                self.accelerator.unwrap_model(self.net).config["out_channels"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
            )
            assert (
                video.shape == expected_video_shape
            ), f"Expected video shape {expected_video_shape}, got {video.shape}"
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    video,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "trajectories",
                    ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
                )

            eval_batches_pbar.update()
            break  # TODO: keep generating on the entire test set and evaluate the trajectories

        # 3. Evaluate the trajectories
        self.logger.warning_once("Should implement evaluation metrics here.")
        # TODO: compute metric on full test data & set self.best_model_to_date accordingly

        eval_batches_pbar.close()
        self.logger.info(
            f"Finished InvertedRegeneration on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

    @log_state(state_logger)
    @torch.inference_mode()
    def _iter_inv_regen(
        self,
        dataloaders: dict[float, DataLoader],
        pbar_manager: Manager,
        eval_strat: IterativeInvertedRegeneration,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        This strategy performs iteratively:
            1. an inversion to obtain the starting Gaussian
            2. a generation from that inverted Gaussian sample to obtain the next image of the video
        over all video timesteps.

        It is thus quite costly to run...

        Note that any other dataloader than the first one is actually not used for evaluation.
        """
        # Checks
        assert (
            list(dataloaders.keys())[0] == 0
        ), f"Expecting the first dataloader to be at time 0, got {list(dataloaders.keys())[0]}"

        # Setup schedulers
        inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inverted_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

        # Use only 1st dataloader for now TODO
        first_dl = list(dataloaders.values())[0]
        eval_batches_pbar = pbar_manager.counter(
            total=len(first_dl),
            position=2,
            desc="Evaluation batches" + 10 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        if self.accelerator.is_main_process:
            eval_batches_pbar.refresh()

        video_times = torch.linspace(0, 1, self.cfg.evaluation.nb_video_timesteps)

        # Now generate & evaluate the trajectories
        for batch_idx, batch in enumerate(iter(first_dl)):
            # 1. Save the to-be inverted images
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    batch,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "starting_samples",
                    ["-1_1 raw", "image min-max"],
                )

            # Generate the trajectory
            # TODO: parallelize the generation along video time?
            # usefull if small inference batch size, otherwise useless
            video = []
            video_time_pbar = pbar_manager.counter(
                total=self.cfg.evaluation.nb_video_timesteps,
                position=3,
                desc="Evaluation video timesteps",
                enable=self.accelerator.is_main_process,
                leave=False,
                min_delta=1,
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            prev_video_time = 0
            image = batch
            for video_t_idx, video_time in enumerate(video_times):
                self.logger.debug(f"Video timestep index {video_t_idx} / {self.cfg.evaluation.nb_video_timesteps}")

                # 2. Generate the inverted Gaussians
                inverted_gauss = image
                inversion_video_time = inference_video_time_encoding.forward(prev_video_time, batch.shape[0])

                for t in inverted_scheduler.timesteps:
                    model_output = self._net_pred(inverted_gauss, t, inversion_video_time, eval_net)  # pyright: ignore[reportArgumentType]
                    inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]

                if batch_idx == 0:
                    save_eval_artifacts_log_to_wandb(
                        inverted_gauss,
                        self.saved_artifacts_folder,
                        self.global_optimization_step,
                        self.accelerator,
                        self.logger,
                        eval_strat.name,
                        "inversions",
                        ["image min-max", "image 5perc-95perc"],
                    )

                # 3. Generate the next image from it
                image = inverted_gauss
                video_time_enc = inference_video_time_encoding.forward(video_time.item(), batch.shape[0])

                for t in inference_scheduler.timesteps:
                    model_output = self._net_pred(image, t, video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                video.append(image.clone())
                prev_video_time = video_time.item()
                video_time_pbar.update()

            video_time_pbar.close(clear=True)

            video = torch.stack(video)
            # wandb expects (batch_size, video_time, channels, height, width)
            video = video.permute(1, 0, 2, 3, 4)
            expected_video_shape = (
                self.cfg.evaluation.batch_size,
                self.cfg.evaluation.nb_video_timesteps,
                self.accelerator.unwrap_model(self.net).config["out_channels"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
            )
            assert (
                video.shape == expected_video_shape
            ), f"Expected video shape {expected_video_shape}, got {video.shape}"
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    video,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "trajectories",
                    ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
                )

            eval_batches_pbar.update()
            break  # TODO: keep generating on the entire test set and evaluate the trajectories

        # 3. Evaluate the trajectories
        self.logger.warning_once("Should implement evaluation metrics here.")
        # TODO: compute metric on full test data & set self.best_model_to_date accordingly

        eval_batches_pbar.close()
        self.logger.info(
            f"Finished IterativeInvertedRegeneration on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

    @log_state(state_logger)
    @torch.inference_mode()
    def _forward_noising(
        self,
        dataloaders: dict[float, DataLoader],
        pbar_manager: Manager,
        eval_strat: ForwardNoising,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        Two steps are performed on each batch of the first dataloader:
            1. Noise the image until eval_start.forward_noising_frac
            2. Generate te trajectory from that slightly noised sample

        Note that any other dataloader than the first one is actually not used for evaluation.
        """
        # Checks
        assert (
            list(dataloaders.keys())[0] == 0
        ), f"Expecting the first dataloader to be at time 0, got {list(dataloaders.keys())[0]}"

        # Setup schedulers: duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

        # Use only 1st dataloader for now TODO
        first_dl = list(dataloaders.values())[0]
        eval_batches_pbar = pbar_manager.counter(
            total=len(first_dl),
            position=2,
            desc="Evaluation batches" + 10 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        if self.accelerator.is_main_process:
            eval_batches_pbar.refresh()

        # Now generate & evaluate the trajectories
        for batch_idx, batch in enumerate(iter(first_dl)):
            # 1. Save the to-be noised images
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    batch,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "starting_samples",
                    ["-1_1 raw", "image min-max"],
                )

            # 2. Sample Gaussian noise and noise the images until some step
            noise = torch.randn(
                batch.shape,
                dtype=batch.dtype,
                device=batch.device,
            )
            noise_timestep_idx = int((1 - eval_strat.forward_noising_frac) * len(inference_scheduler.timesteps))
            noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
            noise_timesteps: IntTensor = torch.full(  # pyright: ignore[reportAssignmentType]
                (batch.shape[0],),
                noise_timestep,
                device=batch.device,
                dtype=torch.int64,
            )
            slightly_noised_sample = self.dynamic.add_noise(batch, noise, noise_timesteps)

            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    slightly_noised_sample,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "noised_samples",
                    ["image min-max", "-1_1 raw", "-1_1 clipped"],
                )

            # 3. Generate the trajectory from it
            # TODO: parallelize the generation along video time?
            # usefull if small inference batch size, otherwise useless
            video = []
            video_time_pbar = pbar_manager.counter(
                total=self.cfg.evaluation.nb_video_timesteps,
                position=3,
                desc="Evaluation video timesteps",
                enable=self.accelerator.is_main_process,
                leave=False,
                min_delta=1,
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            for video_t_idx, video_time in enumerate(torch.linspace(0, 1, self.cfg.evaluation.nb_video_timesteps)):
                image = slightly_noised_sample.clone()
                self.logger.debug(f"Video timestep index {video_t_idx} / {self.cfg.evaluation.nb_video_timesteps}")
                video_time_enc = inference_video_time_encoding.forward(video_time.item(), batch.shape[0])

                for t in inference_scheduler.timesteps[noise_timestep_idx:]:
                    model_output = self._net_pred(image, t, video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                video.append(image)
                video_time_pbar.update()
                # _log_to_file_only(
                #     self.logger,
                #     f"Video time {video_timestep}/{self.cfg.training.eval_nb_video_timesteps}",
                #     INFO,
                # )

            video_time_pbar.close(clear=True)

            video = torch.stack(video)
            # wandb expects (batch_size, video_time, channels, height, width)
            video = video.permute(1, 0, 2, 3, 4)
            expected_video_shape = (
                self.cfg.evaluation.batch_size,
                self.cfg.evaluation.nb_video_timesteps,
                self.accelerator.unwrap_model(self.net).config["out_channels"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
            )
            assert (
                video.shape == expected_video_shape
            ), f"Expected video shape {expected_video_shape}, got {video.shape}"
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    video,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    eval_strat.name,
                    "trajectories",
                    ["image min-max", "video min-max", "-1_1 raw", "-1_1 clipped"],
                )

            eval_batches_pbar.update()
            break  # TODO: keep generating on the entire test set and evaluate the trajectories

        # 3. Evaluate the trajectories
        self.logger.warning_once("Should implement evaluation metrics here.")
        # TODO: compute metric on full test data & set self.best_model_to_date accordingly

        eval_batches_pbar.close()
        self.logger.info(
            f"Finished ForwardNoising on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

    @log_state(state_logger)
    @torch.inference_mode()
    def _fid_computation(
        self,
        tmp_save_folder: Path,
        pbar_manager: Manager,
        eval_strat: FIDComputation,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
        true_data_classes_paths: dict[int, str],
    ):
        """
        Compute FID vs training data.

        Metrics are computed on main process only.
        """
        # 0. Preparations
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

        # use training time encodings
        eval_video_time = torch.tensor(self.empirical_dists_timesteps).to(self.accelerator.device)
        eval_video_time_enc = inference_video_time_encoding.forward(eval_video_time)

        # 1. Generate the samples
        # loop over training video times
        video_times_pbar = pbar_manager.counter(
            total=len(eval_video_time),
            position=2,
            desc="Training video timesteps  ",
            enable=self.accelerator.is_main_process,
            leave=False,
        )
        if self.accelerator.is_main_process:
            video_times_pbar.refresh()

        for video_time_idx, video_time_enc in video_times_pbar(enumerate(eval_video_time_enc)):
            video_time_enc = video_time_enc.unsqueeze(0).repeat(eval_strat.batch_size, 1)

            # find how many samples to generate, batchify generation and distribute along processes
            this_proc_gen_batches = self._find_this_proc_this_time_batches_for_fid_comp(
                eval_strat, video_time_idx, true_data_classes_paths
            )

            batches_pbar = pbar_manager.counter(
                total=len(this_proc_gen_batches),
                position=3,
                desc="Evaluation batch" + 10 * " ",
                enable=self.accelerator.is_main_process,
                leave=False,
            )
            if self.accelerator.is_main_process:
                batches_pbar.refresh()

            # loop over generation batches
            for batch_idx, batch_size in batches_pbar(enumerate(this_proc_gen_batches)):
                gen_pbar = pbar_manager.counter(
                    total=len(inference_scheduler.timesteps),
                    position=4,
                    desc="Generating samples" + " " * 8,
                    enable=self.accelerator.is_main_process,
                    leave=False,
                )
                if self.accelerator.is_main_process:
                    gen_pbar.refresh()

                # generate a batch of samples
                image = torch.randn(
                    batch_size,
                    self.accelerator.unwrap_model(self.net).config["in_channels"],
                    self.accelerator.unwrap_model(self.net).config["sample_size"],
                    self.accelerator.unwrap_model(self.net).config["sample_size"],
                    device=self.accelerator.device,
                )
                video_time_enc = video_time_enc[:batch_size]

                # loop over diffusion timesteps
                for t in gen_pbar(inference_scheduler.timesteps):
                    model_output = self._net_pred(image, t, video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                save_images_for_fid_compute(
                    image,
                    tmp_save_folder / "fid_computation" / str(video_time_idx),
                    sum(this_proc_gen_batches[:batch_idx]),
                    self.accelerator.process_index,
                )

                gen_pbar.close()

            batches_pbar.close()
            # wait for everyone at end of each time (should be enough to avoid timeouts)
            self.accelerator.wait_for_everyone()

        video_times_pbar.close()
        # no need to wait here then
        self.logger.info("Finished image generation")

        # 2. Compute FIDs
        # clear the dataset caches because the dataset might not be exactly the same that in the previous run,
        # despite having the same cfg.dataset.name (used as ID): risk of invalid cache
        # consistency of cache naming with below is important
        caches_to_remove: dict[str | int, Path] = {
            "all_classes": tmp_save_folder / "fid_computation" / self.cfg.dataset.name
        }
        for video_time_idx in range(len(self.empirical_dists_timesteps)):
            caches_to_remove[video_time_idx] = (
                tmp_save_folder / "fid_computation" / (self.cfg.dataset.name + "_class_" + str(video_time_idx))
            )

        if self.accelerator.is_main_process:
            for cache in caches_to_remove.values():
                if cache.exists():
                    shutil.rmtree(cache)
        self.accelerator.wait_for_everyone()

        # TODO: weight tasks by number of samples
        tasks = ["all_classes"] + list(range(len(self.empirical_dists_timesteps)))
        tasks_for_this_process = tasks[self.accelerator.process_index :: self.accelerator.num_processes]

        self.logger.info("Computing FID scores...")
        metrics_dict: dict[str, dict[str, float]] = {}
        for task in tasks_for_this_process:
            if task == "all_classes":
                self.logger.debug(f"Computing metrics against true samples at {Path(self.cfg.dataset.path).as_posix()}")
                metrics = torch_fidelity.calculate_metrics(
                    input1=Path(self.cfg.dataset.path).as_posix(),
                    input2=(tmp_save_folder / "fid_computation").as_posix(),
                    cuda=True,
                    batch_size=eval_strat.batch_size,  # TODO: optimize
                    isc=True,
                    fid=True,
                    prc=True,
                    verbose=self.cfg.debug and self.accelerator.is_main_process,
                    cache_root=(tmp_save_folder / "fid_computation").as_posix(),
                    input1_cache_name=caches_to_remove["all_classes"].name,
                    samples_find_deep=True,
                )
            else:
                assert isinstance(task, int)
                self.logger.debug(f"Computing metrics against true samples at {true_data_classes_paths[task]}")
                metrics = torch_fidelity.calculate_metrics(
                    input1=true_data_classes_paths[task],
                    input2=(tmp_save_folder / "fid_computation" / str(task)).as_posix(),
                    cuda=True,
                    batch_size=eval_strat.batch_size,  # TODO: optimize
                    isc=True,
                    fid=True,
                    prc=True,
                    verbose=self.cfg.debug and self.accelerator.is_main_process,
                    cache_root=(tmp_save_folder / "fid_computation").as_posix(),
                    input1_cache_name=caches_to_remove[task].name,
                )
            metrics_dict[str(task)] = metrics
            self.logger.debug(
                f"Computed metrics for class {task} on process {self.accelerator.process_index}: {metrics}",
                main_process_only=False,
            )
        # save this process' metrics to disk
        with open(tmp_save_folder / f"proc{self.accelerator.process_index}_metrics_dict.pkl", "wb") as pickle_file:
            pickle.dump(metrics_dict, pickle_file)
            self.logger.debug(
                f"Saved metrics for process {self.accelerator.process_index} at {tmp_save_folder}",
                main_process_only=False,
            )
        self.accelerator.wait_for_everyone()

        # 3. Merge metrics from all processes & Log metrics
        if self.accelerator.is_main_process:
            final_metrics_dict = {}
            for metrics_file in [f for f in tmp_save_folder.iterdir() if f.name.endswith("metrics_dict.pkl")]:
                with open(metrics_file, "rb") as pickle_file:
                    proc_metrics = pickle.load(pickle_file)
                    final_metrics_dict.update(proc_metrics)
            self.accelerator.log(
                {
                    f"evaluation/class_{class_idx}/": this_cl_metrics
                    for class_idx, this_cl_metrics in final_metrics_dict.items()
                },
                step=self.global_optimization_step,
            )
            self.logger.info(
                f"Logged metrics {final_metrics_dict}",
            )

    def _find_this_proc_this_time_batches_for_fid_comp(
        self, eval_strat: FIDComputation, video_time_idx: int, true_data_classes_paths: dict[int, str]
    ) -> list[int]:
        # find total number of samples to generate for this video time
        if isinstance(eval_strat.nb_samples_to_gen_per_time, int):
            tot_nb_samples = eval_strat.nb_samples_to_gen_per_time
        elif eval_strat.nb_samples_to_gen_per_time == "adapt":
            tot_nb_samples = len(list(Path(true_data_classes_paths[video_time_idx]).iterdir()))
            self.logger.debug(
                f"Found {tot_nb_samples} samples for class n°{video_time_idx} at {true_data_classes_paths[video_time_idx]}"
            )
        else:
            raise ValueError(
                f"Expected 'nb_samples_to_gen_per_time' to be an int or 'adapt', got {eval_strat.nb_samples_to_gen_per_time}"
            )
        # batchify generation
        nb_full_batches = tot_nb_samples // eval_strat.batch_size
        gen_batches_sizes = [eval_strat.batch_size] * nb_full_batches
        reminder = tot_nb_samples % eval_strat.batch_size
        gen_batches_sizes += [reminder] if reminder != 0 else []
        # share batches between processes
        this_proc_gen_batches = gen_batches_sizes[self.accelerator.process_index :: self.accelerator.num_processes]
        self.logger.debug(
            f"Process {self.accelerator.process_index} will generate batches of sizes: {this_proc_gen_batches}",
            main_process_only=False,
        )
        return this_proc_gen_batches

    @log_state(state_logger)
    def _save_pipeline(self):
        """
        Save the net to disk as an independent pretrained pipeline.

        Can be called by all processes (only main will actually save).
        """
        self.accelerator.unwrap_model(self.net).save_pretrained(
            self.model_save_folder / "net",
            is_main_process=self.accelerator.is_main_process,
        )
        self.accelerator.unwrap_model(self.video_time_encoding).save_pretrained(
            self.model_save_folder / "video_time_encoder",
            is_main_process=self.accelerator.is_main_process,
        )
        if self.accelerator.is_main_process:
            self.dynamic.save_pretrained(self.model_save_folder / "dynamic")
        self.logger.info(f"Saved net, video time encoder and dynamic config to {self.model_save_folder}")

    @log_state(state_logger)
    def _checkpoint(self):
        """
        Save the current state of the models, optimizers, schedulers, and dataloaders.

        Should be called by all processes as `accelerator.save_state` handles checkpointing
        in DDP setting internally, and distributed barriers are used here.

        Used to resume training.
        """
        # First, wait for all processes to reach this point
        self.accelerator.wait_for_everyone()

        this_chkpt_subfolder = self.chckpt_save_path / f"step_{self.global_optimization_step}"
        self.accelerator.save_state(this_chkpt_subfolder.as_posix())

        # Resuming args saved in json file
        training_info_for_resume = {
            "start_instant_batch_idx": self.instant_batch_idx,
            "start_global_optimization_step": self.global_optimization_step,
            "best_model_to_date": self.best_model_to_date,
        }
        if self.accelerator.is_main_process:  # resuming args are common to all processes
            self.logger.info(
                f"Checkpointing resuming args at step {self.global_optimization_step}: {training_info_for_resume}"
            )
            with open(this_chkpt_subfolder / "training_state_info.json", "w", encoding="utf-8") as f:
                json.dump(training_info_for_resume, f)
        self.accelerator.wait_for_everyone()
        # check consistency between processes
        with open(this_chkpt_subfolder / "training_state_info.json", "r", encoding="utf-8") as f:
            assert (
                training_info_for_resume == json.load(f)
            ), f"Expected consistency of resuming args between process {self.accelerator.process_index} and main process; got {training_info_for_resume} != {json.load(f)}"

        # Delete old checkpoints if needed
        if self.accelerator.is_main_process:
            checkpoints_list = [d for d in self.chckpt_save_path.iterdir() if not d.name.startswith(".")]
            nb_checkpoints = len(checkpoints_list)
            if nb_checkpoints > self.cfg.checkpointing.checkpoints_total_limit:
                sorted_chkpt_subfolders = sorted(checkpoints_list, key=lambda x: int(x.name.split("_")[1]))
                to_del = sorted_chkpt_subfolders[: -self.cfg.checkpointing.checkpoints_total_limit]
                if len(to_del) > 1:
                    self.logger.error(f"\033[1;33mMORE THAN 1 CHECKPOINT TO DELETE:\033[0m\n {to_del}")
                for d in to_del:
                    self.logger.info(f"Deleting checkpoint {d.name}...")
                    shutil.rmtree(d)

        self.accelerator.wait_for_everyone()


def resume_from_checkpoint(
    cfg: Config,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
    chckpt_save_path: Path,
) -> ResumingArgs | None:
    """
    Should be called by all processes as `accelerator.load_state` handles checkpointing
    and distributed barriers are used here.
    """
    resume_arg = cfg.checkpointing.resume_from_checkpoint
    # 1. first find the correct subfolder to resume from
    if type(resume_arg) == int:
        path = chckpt_save_path / f"step_{resume_arg}"
    else:
        assert resume_arg is True, "Expected resume_from_checkpoint to be True here"
        # Get the most recent checkpoint
        if not chckpt_save_path.exists() and accelerator.is_main_process:
            logger.warning("No 'checkpoints' directory found in run folder; creating one.")
            chckpt_save_path.mkdir()
        accelerator.wait_for_everyone()
        dirs = [d for d in chckpt_save_path.iterdir() if not d.name.startswith(".")]
        dirs = sorted(dirs, key=lambda d: int(d.name.split("_")[1]))
        path = Path(chckpt_save_path, dirs[-1]) if len(dirs) > 0 else None
    # 2. load state and other resuming args if applicable
    if path is None:
        logger.warning(f"No checkpoint found in {chckpt_save_path}. Starting a new training run.")
        resuming_args = None
    else:
        logger.info(f"Resuming from checkpoint {path}")
        accelerator.load_state(path.as_posix())
        with open(path / "training_state_info.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        resuming_args = ResumingArgs(**data)
        # check consistency
        field_names = [field.name for field in fields(ResumingArgs)]
        assert (
            field_names == list(data.keys())
        ), f"Expected matching field names between ResumingArgs and the loaded JSON file , but got {field_names} and {data.keys()}"

    return resuming_args
