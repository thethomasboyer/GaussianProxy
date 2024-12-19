import json
import pickle
import shutil
from dataclasses import dataclass, field, fields

# from logging import INFO, FileHandler, makeLogRecord
from pathlib import Path
from typing import ClassVar, Generator, Optional, Type

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
    ForwardNoising,
    InvertedRegeneration,
    IterativeInvertedRegeneration,
    MetricsComputation,
    SimpleGeneration,
)
from GaussianProxy.utils.data import BaseDataset
from GaussianProxy.utils.misc import (
    StateLogger,
    get_evenly_spaced_timesteps,
    hard_augment_dataset_all_square_symmetries,
    log_state,
    save_eval_artifacts_log_to_wandb,
    save_images_for_metrics_compute,
)
from GaussianProxy.utils.models import VideoTimeEncoding

# State logger to track time spent in some functions
state_logger = StateLogger()


@dataclass
class ResumingArgs:
    """
    Arguments to resume training from a checkpoint.

    **Must default to "natural" starting values when training from scratch!**
    """

    json_state_filename: ClassVar[str] = "training_state_info.json"  # excluded from fields!

    start_global_optimization_step: int = 0
    # best_metric_to_date must be initialized to the worst possible value
    # Only "smaller is better" metrics are supported!
    best_metric_to_date: float = np.inf

    @classmethod
    def from_dict(cls, data: dict):
        # Check for exact field matching
        assert isinstance(data, dict), f"Expecting a dict, got {type(data)}"
        expected_fields = sorted(f.name for f in fields(cls))
        provided_fields = sorted(data.keys())
        assert expected_fields == provided_fields, f"Expecting fields {expected_fields}, got {provided_fields}"
        return cls(**data)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, path: Path):
        # load the json file
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


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
    _timesteps_names_to_floats: dict[str, float] = field(init=False)
    _timesteps_floats_to_names: dict[float, str] = field(init=False)
    # resuming args
    _resuming_args: ResumingArgs = field(init=False)
    # global training state
    global_optimization_step: int = field(init=False)
    # WARNING: only "smaller is better" metrics are supported!
    # WARNING: best_metric_to_date will only be updated on the main process!
    best_metric_to_date: float = field(init=False)
    first_metrics_eval: bool = True  # always reset to True even if resuming from a checkpoint
    eval_on_start: bool = False
    # constant evaluation starting states
    _eval_noise: Tensor = field(init=False)
    _eval_video_times: Tensor = field(init=False)

    @property
    def nb_empirical_dists(self) -> int:
        if not hasattr(self, "_nb_empirical_dists"):
            raise RuntimeError("nb_empirical_dists is not set; _fit_init should be called before trying to access it")
        return self._nb_empirical_dists

    @property
    def empirical_dists_timesteps(self) -> list[float]:
        if self._empirical_dists_timesteps is None:
            raise RuntimeError(
                "empirical_dists_timesteps is not set; _fit_init should be called before trying to access it"
            )
        return self._empirical_dists_timesteps

    @property
    def resuming_args(self) -> ResumingArgs:
        if self._resuming_args is None:
            raise RuntimeError("resuming_args is not set; _fit_init should be called before trying to access it")
        return self._resuming_args

    @property
    def data_shape(self) -> tuple[int, int, int]:
        if self._data_shape is None:
            raise RuntimeError("data_shape is not set; _fit_init should be called before trying to access it")
        return self._data_shape

    @property
    def timesteps_names_to_floats(self) -> dict[str, float]:
        """dict of empirical distribution names to their float representation"""
        if self._timesteps_names_to_floats is None:
            raise RuntimeError(
                "timesteps_names_to_floats is not set; _fit_init should be called before trying to access it"
            )
        return self._timesteps_names_to_floats

    @property
    def timesteps_floats_to_names(self) -> dict[float, str]:
        """dict of empirical distribution floats to their names representation"""
        if self._timesteps_floats_to_names is None:
            raise RuntimeError(
                "timesteps_floats_to_names is not set; _fit_init should be called before trying to access it"
            )
        return self._timesteps_floats_to_names

    def get_eval_noise(self) -> Tensor:
        if self._eval_noise is None:
            raise RuntimeError("eval_noise is not set; _fit_init should be called before trying to access it")
        return self._eval_noise.clone()

    def get_eval_video_times(self) -> Tensor:
        if self._eval_video_times is None:
            raise RuntimeError("eval_video_times is not set; _fit_init should be called before trying to access it")
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

        Arguments:
            train_dataloaders (dict[int, DataLoader] | dict[str, DataLoader])
            Dictionary of "_fake_"training dataloaders. The actual training batch creation is done in the `_yield_data_batches` method.
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

        # Modify dataloaders dicts to use the empirical distribution timesteps as keys
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

        init_count = self.resuming_args.start_global_optimization_step
        batches_pbar = pbar_manager.counter(
            total=self.cfg.training.nb_time_samplings,
            position=1,
            desc="Training batches" + 10 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            count=init_count,
        )
        if self.accelerator.is_main_process:
            batches_pbar.refresh()
        # loop through training batches
        for time, batch in self._yield_data_batches(
            train_timestep_dataloaders,
            self.logger,
            self.cfg.training.nb_time_samplings - init_count,
        ):
            ### first check if checkpoint or evaluation at *this* opt step (before next gradient step)
            # checkpoint
            if self.global_optimization_step % self.cfg.checkpointing.checkpoint_every_n_steps == 0 and (
                self.global_optimization_step != 0 or self.eval_on_start
            ):
                self._checkpoint()
            # evaluate models every n optimisation steps
            if (
                self.cfg.evaluation.every_n_opt_steps is not None
                and self.global_optimization_step % self.cfg.evaluation.every_n_opt_steps == 0
                and (self.global_optimization_step != 0 or self.eval_on_start)
            ):
                self._evaluate(
                    train_timestep_dataloaders,
                    test_timestep_dataloaders,
                    pbar_manager,
                )
                # reset network to the right mode
                self.net.train()
                self.logger.info(f"Resuming training after evaluation during step {self.global_optimization_step}")

            ### Then train for one gradient step
            # gradient step here
            self._fit_one_batch(batch, time)
            # take one profiler step
            if profiler is not None:
                profiler.step()

            ### Update global opt step and pbar at very end of everything
            self.global_optimization_step += 1
            batches_pbar.update()

        batches_pbar.close(clear=True)
        # update timeline & save it # TODO: broken since epochs removal; to update every n steps (and to fix...)
        gantt_chart = state_logger.create_gantt_chart()
        self.accelerator.log({"state_timeline/": wandb.Plotly(gantt_chart)})

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
        """
        Fit some remaining attributes before fitting.

        Any other method than `load_checkpoint` should be called *after* this method.
        """
        assert not hasattr(self, "._nb_empirical_dists"), "Already fitted"
        self._nb_empirical_dists = len(train_dataloaders)
        assert self._nb_empirical_dists > 1, "Expecting at least 2 empirical distributions to train the model."
        self._empirical_dists_timesteps = get_evenly_spaced_timesteps(self.nb_empirical_dists)
        # save original timestamp to float timestamp mapping
        self._timesteps_names_to_floats = {
            str(tp_name): self.empirical_dists_timesteps[idx] for idx, tp_name in enumerate(train_dataloaders.keys())
        }
        self._timesteps_floats_to_names = {v: k for k, v in self.timesteps_names_to_floats.items()}
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
        self.best_metric_to_date = self.resuming_args.best_metric_to_date
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
                    # DataLoader iterators will quickly be exhausted (resulting in silent hangs then undebuggable NCCL timeouts 🙃)
                    # so reform them when needed
                    except StopIteration:
                        self.logger.debug(
                            f"Reforming dataloader iterator for timestep {t_to_sample_from} on process {self.accelerator.process_index}",
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
                    level=wandb.AlertLevel.WARN,  # pyright: ignore[reportAttributeAccessIssue]
                    wait_duration=21600,  # 6 hours
                )
                self.logger.error(msg)
                # TODO: restart from previous checkpoint if is NaN repeatedly?

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
            }
        )

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
        raw_train_dataloaders: dict[float, DataLoader[Tensor]],
        test_dataloaders: dict[float, DataLoader[Tensor]],
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
        # to ensure models are "clean" are results perfectly reproducible
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
        # compile inference model
        if self.cfg.accelerate.launch_args.dynamo_backend != "no":
            inference_net = torch.compile(inference_net)  # pyright: ignore[reportAssignmentType]
        # TODO: save & load a compiled artifact when torch.export is stabilized

        # 3. Run through evaluation strategies
        for eval_strat in self.cfg.evaluation.strategies:  # TODO: match on type when config is updated
            if eval_strat.name == "SimpleGeneration":
                self._simple_gen(
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )

            elif eval_strat.name == "ForwardNoising":
                self._forward_noising(
                    test_dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )

            elif eval_strat.name == "InvertedRegeneration":
                # get first batches of train and test time-0 dataloaders
                train_dl_time0 = raw_train_dataloaders[0]
                train_batch_time0: Tensor = next(iter(train_dl_time0))[: self.cfg.training.train_batch_size]
                while train_batch_time0.shape[0] != self.cfg.training.train_batch_size:
                    train_batch_time0 = torch.cat([train_batch_time0, next(iter(train_dl_time0))])
                    train_batch_time0 = train_batch_time0[: self.cfg.training.train_batch_size]
                assert (
                    expected_shape := (
                        self.cfg.training.train_batch_size,
                        *self.data_shape,
                    )
                ) == train_batch_time0.shape, f"Expected shape {expected_shape}, got {train_batch_time0.shape}"

                test_dl_time0 = list(test_dataloaders.values())[0]
                test_batch_time0: Tensor = next(iter(test_dl_time0))
                assert (
                    expected_shape := (
                        self.cfg.evaluation.batch_size,
                        *self.data_shape,
                    )
                ) == test_batch_time0.shape, f"Expected shape {expected_shape}, got {test_batch_time0.shape}"

                # run the evaluation strategy on theses
                self._inv_regen(
                    train_batch_time0,
                    test_batch_time0,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )

            elif eval_strat.name == "IterativeInvertedRegeneration":
                self._iter_inv_regen(
                    test_dataloaders,
                    pbar_manager,
                    eval_strat,  # pyright: ignore[reportArgumentType]
                    inference_net,
                    inference_video_time_encoding,
                )

            elif eval_strat.name == "MetricsComputation":
                true_data_classes_paths: dict[str, str] = {}  # time names to str paths
                for time_float, dl in test_dataloaders.items():
                    inner_ds = dl.dataset
                    if not isinstance(inner_ds, BaseDataset):
                        raise ValueError(
                            f"Expected a `BaseDataset` for the underlying dataset of the evaluation dataloader, got {type(inner_ds)}"
                        )
                    # assuming they all share the same parent...
                    this_class_path = Path(inner_ds.samples[0]).parent
                    tp_name = self.timesteps_floats_to_names[time_float]
                    if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":  # pyright: ignore[reportAttributeAccessIssue]
                        # use the hard augmented version of the dataset if it's not the one we are already training on
                        if "_hard_augmented" not in this_class_path.parent.name:
                            hard_augmented_dataset_path = (
                                this_class_path.parent.with_name(this_class_path.parent.name + "_hard_augmented")
                                / this_class_path.name
                            )
                            self.logger.debug(
                                f"Using hard augmented version of {this_class_path}: {hard_augmented_dataset_path}"
                            )
                        else:
                            hard_augmented_dataset_path = this_class_path
                        # check if it exists
                        assert (
                            hard_augmented_dataset_path.exists()
                        ), f"hard augmented dataset does not exist at {hard_augmented_dataset_path} "
                        true_data_classes_paths[tp_name] = hard_augmented_dataset_path.as_posix()
                    else:
                        true_data_classes_paths[tp_name] = this_class_path.as_posix()
                self._metrics_computation(
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

    @log_state(state_logger)
    @torch.inference_mode()
    def _simple_gen(
        self,
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
            ["-1_1 raw"],
            captions=[f"time: {round(t.item(), 3)}" for t in random_video_time],
        )

        gen_pbar.close(clear=True)

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
        train_batch_time_zero: Tensor,
        test_batch_time_zero: Tensor,
        pbar_manager: Manager,
        eval_strat: InvertedRegeneration,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
    ):
        """
        A quick visualization test that generates a (small) batch of videos, both on train and test starting data.
        Starting data *must* be of time zero.

        Two steps are performed on the 2 batches:
            1. Perform inversion to obtain the starting Gaussian
            2. Generate te trajectory from that inverted Gaussian sample
        """
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

        eval_batches_pbar = pbar_manager.counter(
            total=2,
            position=2,
            desc="Evaluation batches" + 8 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        if self.accelerator.is_main_process:
            eval_batches_pbar.refresh()

        # Now generate & evaluate the trajectories
        for split, batch in [
            ("train_split", train_batch_time_zero),
            ("test_split", test_batch_time_zero),
        ]:
            # 1. Save the to-be inverted images
            save_eval_artifacts_log_to_wandb(
                batch,
                self.saved_artifacts_folder,
                self.global_optimization_step,
                self.accelerator,
                self.logger,
                eval_strat.name,
                "starting_samples",
                ["-1_1 raw"],
                split_name=split,
            )

            # 2. Generate the inverted Gaussians
            inverted_gauss = batch
            inversion_video_time = inference_video_time_encoding.forward(0, batch.shape[0])

            for t in inverted_scheduler.timesteps:
                model_output = self._net_pred(inverted_gauss, t, inversion_video_time, eval_net)  # pyright: ignore[reportArgumentType]
                inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]

            save_eval_artifacts_log_to_wandb(
                inverted_gauss,
                self.saved_artifacts_folder,
                self.global_optimization_step,
                self.accelerator,
                self.logger,
                eval_strat.name,
                "inversions",
                ["image min-max", "image 5perc-95perc"],
                split_name=split,
            )

            # 3. Regenerate the starting samples from their inversion
            regen = inverted_gauss.clone()
            for t in inference_scheduler.timesteps:
                model_output = self._net_pred(regen, t, inversion_video_time, eval_net)  # pyright: ignore[reportArgumentType]
                regen = inference_scheduler.step(model_output, int(t), regen, return_dict=False)[0]
            save_eval_artifacts_log_to_wandb(
                regen,
                self.saved_artifacts_folder,
                self.global_optimization_step,
                self.accelerator,
                self.logger,
                eval_strat.name,
                "regenerations",
                ["-1_1 raw"],
                split_name=split,
            )

            # 4. Generate the trajectory from it
            # TODO: parallelize the generation along video time
            video = []
            video_time_pbar = pbar_manager.counter(
                total=self.cfg.evaluation.nb_video_timesteps,
                position=3,
                desc="Evaluation video timesteps",
                enable=self.accelerator.is_main_process,
                leave=False,
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            for video_time in torch.linspace(0, 1, self.cfg.evaluation.nb_video_timesteps):
                image = inverted_gauss.clone()
                video_time_enc = inference_video_time_encoding.forward(video_time.item(), batch.shape[0])

                for t in inference_scheduler.timesteps:
                    model_output = self._net_pred(image, t, video_time_enc, eval_net)  # pyright: ignore[reportArgumentType]
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                video.append(image)
                video_time_pbar.update()

            video_time_pbar.close(clear=True)

            video = torch.stack(video)
            # wandb expects (batch_size, video_time, channels, height, width)
            video = video.permute(1, 0, 2, 3, 4)
            expected_video_shape = (
                batch.shape[0],
                self.cfg.evaluation.nb_video_timesteps,
                self.accelerator.unwrap_model(self.net).config["out_channels"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
                self.accelerator.unwrap_model(self.net).config["sample_size"],
            )
            assert (
                expected_video_shape == video.shape
            ), f"Expected video shape {expected_video_shape}, got {video.shape}"
            save_eval_artifacts_log_to_wandb(
                video,
                self.saved_artifacts_folder,
                self.global_optimization_step,
                self.accelerator,
                self.logger,
                eval_strat.name,
                "trajectories",
                ["-1_1 raw"],
                split_name=split,
            )
            eval_batches_pbar.update()

            # wait for everyone between each split
            self.accelerator.wait_for_everyone()

        eval_batches_pbar.close(clear=True)
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
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            prev_video_time = 0
            image = batch
            for video_t_idx, video_time in enumerate(video_times):
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
            break  # TODO: clean this up either by making this smol-batch-logging and duplicating it on train data

        eval_batches_pbar.close(clear=True)
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
            )
            if self.accelerator.is_main_process:
                video_time_pbar.refresh()

            for video_t_idx, video_time in enumerate(torch.linspace(0, 1, self.cfg.evaluation.nb_video_timesteps)):
                image = slightly_noised_sample.clone()
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
            break  # TODO: clean this up either by making this smol-batch-logging and duplicating it on train data

        eval_batches_pbar.close(clear=True)
        self.logger.info(
            f"Finished ForwardNoising on process ({self.accelerator.process_index})",
            main_process_only=False,
        )

    @log_state(state_logger)
    @torch.inference_mode()
    def _metrics_computation(
        self,
        tmp_save_folder: Path,
        pbar_manager: Manager,
        eval_strat: MetricsComputation,
        eval_net: UNet2DModel | UNet2DConditionModel,
        inference_video_time_encoding: VideoTimeEncoding,
        true_data_classes_paths: dict[str, str],
    ):
        """
        Compute metrics such as FID.

        Everything is distributed.
        """
        ##### 0. Preparations
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # pyright: ignore[reportAssignmentType]
        inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

        # Misc.
        self.logger.info(f"Starting {eval_strat.name}: {eval_strat}")
        self.logger.debug(
            f"Starting {eval_strat.name} on process ({self.accelerator.process_index})",
            main_process_only=False,
        )
        metrics_computation_folder = tmp_save_folder / "metrics_computation"
        assert (
            len(self.empirical_dists_timesteps) == len(true_data_classes_paths)
        ), f"Mismatch between number of timesteps and classes: empirical_dists_timesteps={self.empirical_dists_timesteps}, true_data_classes_paths={true_data_classes_paths}"
        if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
            assert all(
                "hard_augmented" in true_data_classes_paths[key] for key in true_data_classes_paths
            ), f"Expected all true data paths to be hard augmented, got:\n{true_data_classes_paths}"

        # use training time encodings
        eval_video_time = torch.tensor(self.empirical_dists_timesteps).to(self.accelerator.device)
        eval_video_time_enc = inference_video_time_encoding.forward(eval_video_time)

        ##### 1. Generate the samples
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

            # get timestep name
            video_time_name = self.timesteps_floats_to_names[self.empirical_dists_timesteps[video_time_idx]]

            # find how many samples to generate, batchify generation and distribute along processes
            gen_dir = metrics_computation_folder / video_time_name
            gen_dir.mkdir(parents=True, exist_ok=True)
            this_proc_gen_batches = self._find_this_proc_this_time_batches_for_metrics_comp(
                eval_strat,
                Path(true_data_classes_paths[video_time_name]),
                gen_dir,
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
            for batch_size in batches_pbar(this_proc_gen_batches):
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

                save_images_for_metrics_compute(
                    image,
                    gen_dir,
                    self.accelerator.process_index,
                )

                gen_pbar.close(clear=True)

            batches_pbar.close(clear=True)
            # wait for everyone at end of each time (should be enough to avoid timeouts)
            self.accelerator.wait_for_everyone()

        video_times_pbar.close(clear=True)
        # no need to wait here then
        self.logger.debug("Finished image generation in MetricsComputation")

        ##### 1.5 Augment the generated samples if applicable
        if eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
            # on main process:
            if self.accelerator.is_main_process:
                # augment
                extension = "png"  # TODO: remove this hardcoded extension (by moving DatasetParams'params into the base DataSet class used in config, another TODO)
                hard_augment_dataset_all_square_symmetries(
                    metrics_computation_folder,
                    self.logger,
                    extension,
                )
                # check result
                subdirs = [d for d in metrics_computation_folder.iterdir() if d.is_dir()]
                nb_elems_per_class = {
                    class_path.name: len(list((metrics_computation_folder / class_path.name).glob(f"*.{extension}")))
                    for class_path in subdirs
                }
                assert all(
                    nb_elems_per_class[cl_name] % 8 == 0 for cl_name in nb_elems_per_class
                ), f"Expected number of elements to be a multiple of 8, got:\n{nb_elems_per_class}"

        self.accelerator.wait_for_everyone()

        ##### 2. Compute metrics
        # consistency of cache naming with below is important
        metrics_caches: dict[str, Path] = {"all_classes": metrics_computation_folder / self.cfg.dataset.name}
        for video_time_names in list(self.timesteps_names_to_floats.keys()):
            metrics_caches[video_time_names] = metrics_computation_folder / (
                self.cfg.dataset.name + "_time_" + video_time_names
            )

        # clear the dataset caches (on first eval of a run only...)
        # because the dataset might not be exactly the same that in the previous run,
        # despite having the same cfg.dataset.name (used as ID): risk of invalid cache!
        if self.first_metrics_eval:
            self.first_metrics_eval = False
            # clear on main process
            self.logger.debug("Clearing dataset caches for metrics computation")
            if self.accelerator.is_main_process:
                for cache in metrics_caches.values():
                    if cache.exists():
                        shutil.rmtree(cache)
        self.accelerator.wait_for_everyone()

        # TODO: weight tasks by number of samples
        # TODO: include "all_classes" in the tasks, but ***differentiate between using seen data only and all the available dataset!***
        # tasks = ["all_classes"] + list(self.timesteps_names_to_floats.keys())
        tasks = list(self.timesteps_names_to_floats.keys())  # time names
        tasks_for_this_process = tasks[self.accelerator.process_index :: self.accelerator.num_processes]
        self.logger.debug(
            f"Tasks for process {self.accelerator.process_index}: {tasks_for_this_process}", main_process_only=False
        )

        self.logger.info("Computing metrics...")
        metrics_dict: dict[str, dict[str, float]] = {}
        for task in tasks_for_this_process:
            if task == "all_classes":
                true_samples_path = Path(self.cfg.dataset.path)
                self.logger.debug(
                    f"Computing metrics against {len(list(true_samples_path.iterdir()))} true samples at {true_samples_path.as_posix()} on process {self.accelerator.process_index}",
                    main_process_only=False,
                )
                metrics = torch_fidelity.calculate_metrics(
                    input1=true_samples_path.as_posix(),
                    input2=metrics_computation_folder.as_posix(),
                    cuda=True,
                    batch_size=eval_strat.batch_size * 4,  # TODO: optimize
                    isc=False,
                    fid=True,
                    prc=False,
                    verbose=self.cfg.debug and self.accelerator.is_main_process,
                    cache_root=metrics_computation_folder.as_posix(),
                    input1_cache_name=metrics_caches["all_classes"].name,
                    samples_find_deep=True,
                )
            else:
                self.logger.debug(
                    f"Computing metrics against {len(list(Path(true_data_classes_paths[task]).iterdir()))} true samples at {true_data_classes_paths[task]} on process {self.accelerator.process_index}",
                    main_process_only=False,
                )
                metrics = torch_fidelity.calculate_metrics(
                    input1=true_data_classes_paths[task],
                    input2=(metrics_computation_folder / task).as_posix(),
                    cuda=True,
                    batch_size=eval_strat.batch_size * 4,  # TODO: optimize
                    isc=False,
                    fid=True,
                    prc=False,
                    verbose=self.cfg.debug and self.accelerator.is_main_process,
                    cache_root=metrics_computation_folder.as_posix(),
                    input1_cache_name=metrics_caches[task].name,
                )
            metrics_dict[task] = metrics
            self.logger.debug(
                f"Computed metrics for time {task} on process {self.accelerator.process_index}: {metrics}",
                main_process_only=False,
            )
        # save this process' metrics to disk
        with open(
            tmp_save_folder / f"proc{self.accelerator.process_index}_metrics_dict.pkl",
            "wb",
        ) as pickle_file:
            pickle.dump(metrics_dict, pickle_file)
            self.logger.debug(
                f"Saved metrics for process {self.accelerator.process_index} at {tmp_save_folder}",
                main_process_only=False,
            )
        self.accelerator.wait_for_everyone()

        ##### 3. Merge metrics from all processes & Log metrics
        if self.accelerator.is_main_process:
            final_metrics_dict: dict[str, dict[str, float]] = {}
            # load all processes' metrics
            for metrics_file in [f for f in tmp_save_folder.iterdir() if f.name.endswith("metrics_dict.pkl")]:
                with open(metrics_file, "rb") as pickle_file:
                    proc_metrics = pickle.load(pickle_file)
                    final_metrics_dict.update(proc_metrics)
            # log it
            self.accelerator.log(
                {
                    f"evaluation/{time_name}/": this_time_metrics
                    for time_name, this_time_metrics in final_metrics_dict.items()
                }
            )
            self.logger.info(
                f"Logged metrics {final_metrics_dict}",
            )

        ##### 4. Check if best model to date
        if self.accelerator.is_main_process:
            # WARNING: only "smaller is better" metrics are supported!
            # WARNING: best_metric_to_date will only be updated on the main process!
            if "all_classes" not in final_metrics_dict:  # pyright: ignore[reportPossiblyUnboundVariable]
                key = list(final_metrics_dict.keys())[0]  # pyright: ignore[reportPossiblyUnboundVariable]
                assert isinstance(key, str), f"Expected a string key, got {key} of type {type(key)}"
                self.logger.warning(
                    f"No 'all_classes' key in final_metrics_dict, will update best_metric_to_date with the FID of the first class ({key})"
                )
            else:
                key = "all_classes"
            if (
                self.cfg.debug or self.best_metric_to_date > final_metrics_dict[key]["frechet_inception_distance"]  # pyright: ignore[reportPossiblyUnboundVariable]
            ):
                self.best_metric_to_date = final_metrics_dict[key]["frechet_inception_distance"]  # pyright: ignore[reportPossiblyUnboundVariable]
                self.accelerator.log({"training/best_metric_to_date": self.best_metric_to_date})
                self.logger.info(f"Saving best model to date with class='{key}' FID: {self.best_metric_to_date}")
                self._save_pipeline(called_on_main_process_only=True)

        ##### 5. Clean up (important because we reuse existing generated samples!)
        if self.accelerator.is_main_process:
            shutil.rmtree(metrics_computation_folder)
            self.logger.debug(f"Cleaned up metrics computation folder {metrics_computation_folder}")

    def _find_this_proc_this_time_batches_for_metrics_comp(
        self,
        eval_strat: MetricsComputation,
        true_data_class_path: Path,
        gen_dir: Path,
    ) -> list[int]:
        """
        Return the list of batch sizes to generate, for a given `video_time_idx` and splitting between processes.

        `eval_strat.nb_samples_to_gen_per_time` can be:
        - an `int`: the number of samples to generate
        - `"adapt"`: generate as many samples as there are in the true data time
        - `"adapt half"`: generate half as many samples as there are in the true data time
        - `"adapt half aug"`: generate half as many samples as there are in the true data time, then 8⨉ augment them (Dih4)
        """
        # find total number of samples to generate for this video time
        if isinstance(eval_strat.nb_samples_to_gen_per_time, int):
            tot_nb_samples = eval_strat.nb_samples_to_gen_per_time
        elif eval_strat.nb_samples_to_gen_per_time == "adapt":
            tot_nb_samples = len(list(true_data_class_path.iterdir()))
            self.logger.debug(
                f"Will generate {tot_nb_samples} samples for time {true_data_class_path.name} at {true_data_class_path.as_posix()}"
            )
        elif eval_strat.nb_samples_to_gen_per_time == "adapt half":
            tot_nb_samples = len(list(true_data_class_path.iterdir())) // 2
            self.logger.debug(
                f"Will generate {tot_nb_samples} samples for time {true_data_class_path.name} at {true_data_class_path.as_posix()}"
            )
        elif eval_strat.nb_samples_to_gen_per_time == "adapt half aug":
            nb_all_aug_samples = len(list(true_data_class_path.iterdir()))
            assert (
                nb_all_aug_samples % 8 == 0
            ), f"Expected number of samples to be a multiple of 8, got {nb_all_aug_samples}"
            tot_nb_samples = nb_all_aug_samples // (8 * 2)  # half the number of samples, *then* 8⨉ augment them
            self.logger.debug(
                f"Will generate {tot_nb_samples} samples for time {true_data_class_path.name} at {true_data_class_path.as_posix()}"
            )
            self.logger.debug("Will augment samples 8⨉ after generation")
        else:
            raise ValueError(
                f"Expected 'nb_samples_to_gen_per_time' to be an int, 'adapt', 'adapt half', or 'adapt half aug', got {eval_strat.nb_samples_to_gen_per_time}"
            )

        # Resume from interrupted generation
        already_existing_samples = len(list(gen_dir.iterdir()))
        self.logger.debug(
            f"Found {already_existing_samples} already existing samples for time {true_data_class_path.name} at {gen_dir}"
        )
        tot_nb_samples -= already_existing_samples

        # share equally among processes & batchify
        # the code below ensures all processes have the same number of batches to generate,
        # with the same number of samples in each batch but the last one
        # (with a diff of at most 1 for the last)
        this_proc_nb_full_batches, last_batch_to_share = divmod(
            tot_nb_samples, self.accelerator.num_processes * eval_strat.batch_size
        )
        this_proc_last_batch, remainder = divmod(last_batch_to_share, self.accelerator.num_processes)
        this_proc_gen_batches = [eval_strat.batch_size] * this_proc_nb_full_batches + [this_proc_last_batch]
        if self.accelerator.process_index < remainder:
            this_proc_gen_batches[-1] += 1
        # pop the last batch if it is empty
        if this_proc_gen_batches[-1] == 0:
            this_proc_gen_batches.pop()

        self.logger.debug(
            f"Process {self.accelerator.process_index} will generate batches of sizes: {this_proc_gen_batches} for time {true_data_class_path.name}",
            main_process_only=False,
        )
        return this_proc_gen_batches

    @log_state(state_logger)
    def _save_pipeline(self, called_on_main_process_only: bool = False):
        """
        Save the net, time encoder, and dynamic config to disk as an independent pretrained pipeline.
        Also save the current ResumingArgs to that same folder for later "model save" resuming.

        Can be called by all processes (only main will actually save), or by main only (then no barrier).
        """
        # net
        self.accelerator.unwrap_model(self.net).save_pretrained(
            self.model_save_folder / "net",
            is_main_process=self.accelerator.is_main_process,
        )
        # video time encoder
        self.accelerator.unwrap_model(self.video_time_encoding).save_pretrained(
            self.model_save_folder / "video_time_encoder",
            is_main_process=self.accelerator.is_main_process,
        )
        if self.accelerator.is_main_process:
            # dynamic config
            self.dynamic.save_pretrained(self.model_save_folder / "dynamic")
            # resuming args
            training_info_for_resume = {
                "start_global_optimization_step": self.global_optimization_step,
                "best_metric_to_date": self.best_metric_to_date,
            }
            resuming_args = ResumingArgs.from_dict(training_info_for_resume)
            resuming_args.to_json(self.model_save_folder / ResumingArgs.json_state_filename)

        if not called_on_main_process_only:
            self.accelerator.wait_for_everyone()

        self.logger.info(
            f"Saved net, video time encoder, dynamic config, and resuming args to {self.model_save_folder}"
        )

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
            "start_global_optimization_step": self.global_optimization_step,
            "best_metric_to_date": self.best_metric_to_date,
        }
        if self.accelerator.is_main_process:  # resuming args are common to all processes
            self.logger.info(
                f"Checkpointing resuming args at step {self.global_optimization_step}: {training_info_for_resume}"
            )
            # create an instance before saving to check consistency
            resuming_args = ResumingArgs.from_dict(training_info_for_resume)
            resuming_args.to_json(this_chkpt_subfolder / ResumingArgs.json_state_filename)
        self.accelerator.wait_for_everyone()

        # check consistency between processes
        with open(
            this_chkpt_subfolder / ResumingArgs.json_state_filename,
            "r",
            encoding="utf-8",
        ) as f:
            assert (
                training_info_for_resume == (reloaded_json := json.load(f))
            ), f"Expected consistency of resuming args between process {self.accelerator.process_index} and main process; got {training_info_for_resume} != {reloaded_json}"
        self.accelerator.wait_for_everyone()  # must wait here as we *could* delete the just-saved checkpoint (see warning msg in misc/create_repo_structure)

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

    @log_state(state_logger)
    def load_checkpoint(self, resuming_path: str, logger: MultiProcessAdapter):
        """
        Load a checkpoint saved with `save_state`.

        `accelerator.prepare` must have been called before.
        """
        assert (
            self.cfg.checkpointing.resume_from_checkpoint != "model_save"
        ), "Cannot load a 'model save' checkpoint here"
        # no self.logger here as fit_init has not yet been called!
        logger.info(f"Loading checkpoint and training state from {resuming_path}")
        # "automatic" resuming from some save_state checkpoint
        self.accelerator.load_state(resuming_path)


def load_resuming_args(
    cfg: Config,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
    chckpt_save_path: Path,
    models_save_folder: Path,
):
    """
    Handles the location of the correct checkpoint folder to resume training from,
    and loads ResumingArgs from it (if applicable).

    Must be called by all processes.
    """
    resume_arg = cfg.checkpointing.resume_from_checkpoint
    # 1. first find the correct subfolder to resume from;
    # 3 possibilities:
    # - int: resume from a specific step
    # - True: resume from the most recent checkpoint
    # - "model_save": resume from the last *saved model* (!= checkpoint!)
    if type(resume_arg) == int:  # noqa E721
        resuming_path = chckpt_save_path / f"step_{resume_arg}"
    elif resume_arg is True:
        # Get the most recent checkpoint
        if not chckpt_save_path.exists() and accelerator.is_main_process:
            logger.warning("No 'checkpoints' directory found in run folder; creating one.")
            chckpt_save_path.mkdir()
        accelerator.wait_for_everyone()
        dirs = [d for d in chckpt_save_path.iterdir() if not d.name.startswith(".")]
        dirs = sorted(dirs, key=lambda d: int(d.name.split("_")[1]))
        resuming_path = Path(chckpt_save_path, dirs[-1]) if len(dirs) > 0 else None
    elif resume_arg == "model_save":
        resuming_path = models_save_folder
    else:
        raise ValueError(
            f"Expected 'checkpointing.resume_from_checkpoint' to be an int, True or 'model_save' at this point, got {resume_arg}"
        )
    # 2. return resuming args if applicable
    if resuming_path is None:
        logger.warning(f"No checkpoint found in {chckpt_save_path}. Starting a new training run.")
        resuming_args = None
    else:
        logger.info(f"Loading resuming args from {resuming_path}")
        resuming_args = ResumingArgs.from_json(resuming_path / ResumingArgs.json_state_filename)
        logger.info(f"Loaded resuming arguments: {resuming_args}")
    return resuming_path, resuming_args
