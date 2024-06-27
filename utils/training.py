import json
import os
import shutil
from dataclasses import dataclass, field, fields

# from logging import INFO, FileHandler, makeLogRecord
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from enlighten import Manager, get_manager
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.profiler import profile as torch_profile
from torch.profiler import schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

from conf.conf import Checkpointing, Config, Training
from utils.misc import get_evenly_spaced_timesteps, save_eval_artifacts_log_to_wandb
from utils.models import VideoTimeEncoding


@dataclass
class ResumingArgs:
    """
    Arguments to resume training from a checkpoint.

    Must default to "natural" starting values when training from scratch.
    """

    start_instant_batch_idx: int = 0
    start_global_optimization_step: int = 0
    start_global_epoch: int = 0
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
    dynamic: DDIMScheduler
    net: UNet2DModel | UNet2DConditionModel
    video_time_encoding: VideoTimeEncoding
    accelerator: Accelerator
    debug: bool
    # populated arguments when calling .fit
    _training_cfg: Training = field(init=False)
    _checkpointing_cfg: Checkpointing = field(init=False)
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
    global_epoch: int = field(init=False)
    global_optimization_step: int = field(init=False)
    best_model_to_date: bool = True  # TODO

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
    def training_cfg(self) -> Training:
        if self._training_cfg is None:
            raise RuntimeError("training_cfg is not set; fit should be called before trying to access it")
        return self._training_cfg

    @property
    def checkpointing_cfg(self) -> Checkpointing:
        if self._checkpointing_cfg is None:
            raise RuntimeError("checkpointing_cfg is not set; fit should be called before trying to access it")
        return self._checkpointing_cfg

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

    def fit(
        self,
        train_dataloaders: dict[int, DataLoader],
        test_dataloaders: dict[int, DataLoader],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        logger: MultiProcessAdapter,
        output_dir: str,
        model_save_folder: Path,
        saved_artifacts_folder: Path,
        training_cfg: Training,
        checkpointing_cfg: Checkpointing,
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
            training_cfg,
            checkpointing_cfg,
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
            assert np.all(
                np.diff(list(dls.keys())) > 0
            ), "Expecting original dataloaders to be numbered in increasing order."
            timestep_dataloaders = train_timestep_dataloaders if split == "train" else test_timestep_dataloaders
            for dataloader_idx, dl in enumerate(dls.values()):
                timestep_dataloaders[self.empirical_dists_timesteps[dataloader_idx]] = dl
            assert (
                len(timestep_dataloaders) == self.nb_empirical_dists == len(dls)
            ), f"Got {len(timestep_dataloaders)} dataloaders, nb_empirical_dists={self.nb_empirical_dists} and len(dls)={len(dls)}; they should be equal"
            assert np.all(
                np.diff(list(dls.keys())) > 0
            ), "Expecting newly key-ed dataloaders to be numbered in increasing order."

        pbar_manager: Manager = get_manager()  # pyright: ignore[reportAssignmentType]

        epochs_pbar = pbar_manager.counter(
            total=self.training_cfg.nb_epochs,
            desc="Epochs" + 32 * " ",
            position=1,
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
            count=self.resuming_args.start_global_epoch,
        )
        epochs_pbar.refresh()

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

        for epoch in range(self.resuming_args.start_global_epoch, self.training_cfg.nb_epochs):
            self.epoch = epoch
            self._fit_epoch(
                train_timestep_dataloaders,
                test_timestep_dataloaders,
                pbar_manager,
                profiler,
            )
            epochs_pbar.update()
            # evaluate models every n epochs
            if self.training_cfg.eval_every_n_epochs is not None and epoch % self.training_cfg.eval_every_n_epochs == 0:
                self._evaluate(
                    test_timestep_dataloaders,
                    pbar_manager,
                )
            self.accelerator.wait_for_everyone()
        if profiler is not None:
            profiler.stop()

    def _fit_init(
        self,
        train_dataloaders: dict[int, DataLoader],
        training_cfg: Training,
        checkpointing_cfg: Checkpointing,
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
        self._training_cfg = training_cfg
        self._checkpointing_cfg = checkpointing_cfg
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
        self.global_epoch = self.resuming_args.start_global_epoch
        self.best_model_to_date = self.resuming_args.best_model_to_date
        # expected data shape
        unwrapped_net_config = self.accelerator.unwrap_model(self.net).config
        self._data_shape = (
            unwrapped_net_config["in_channels"],
            unwrapped_net_config["sample_size"],
            unwrapped_net_config["sample_size"],
        )

    def _fit_epoch(
        self,
        train_dataloaders: dict[float, DataLoader],
        test_dataloaders: dict[float, DataLoader],
        pbar_manager: Manager,
        profiler: Optional[torch_profile] = None,
    ):
        """
        Fit a whole epoch of data.
        """
        init_count = (
            self.resuming_args.start_instant_batch_idx if self.epoch == self.resuming_args.start_global_epoch else 0
        )
        batches_pbar = pbar_manager.counter(
            total=self.training_cfg.nb_time_samplings,
            position=2,
            desc="Iterating over training data batches" + 2 * " ",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
            count=init_count,
        )
        batches_pbar.refresh()
        # loop through training batches
        for batch_idx, (time, batch) in enumerate(
            self._yield_data_batches(train_dataloaders, self.logger, self.training_cfg.nb_time_samplings - init_count)
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
            if self.global_optimization_step % self.checkpointing_cfg.checkpoint_every_n_steps == 0:
                self._checkpoint()
            # update pbar
            batches_pbar.update()
            # evaluate models every n optimisation steps
            if (
                self.training_cfg.eval_every_n_opt_steps is not None
                and self.global_optimization_step % self.training_cfg.eval_every_n_opt_steps == 0
            ):
                batches_pbar.close()  # close otherwise interference w/ evaluation pbar
                self._evaluate(
                    test_dataloaders,
                    pbar_manager,
                )
                batches_pbar = pbar_manager.counter(
                    total=self.training_cfg.nb_time_samplings,
                    position=2,
                    desc="Iterating over training data batches" + 2 * " ",
                    enable=self.accelerator.is_main_process,
                    leave=False,
                    count=self.instant_batch_idx,
                    min_delta=1,
                )
            self.accelerator.wait_for_everyone()
        batches_pbar.close()
        # update epoch
        self.global_epoch += 1

    @torch.no_grad()
    def _yield_data_batches(
        self, dataloaders: dict[float, DataLoader], logger: MultiProcessAdapter, nb_time_samplings: int
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
        continuous time... Along many epochs the theoretical end result will of course be the same.
        """
        # TODO: not actually needed! Just here for sanity check but actually this assert is wrong
        first_dl = list(dataloaders.values())[0]
        assert all(
            len(first_dl) == len(dl) for dl in dataloaders.values()
        ), "All dataloaders should have the same length"
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
            random_sampling = torch.rand((self.training_cfg.train_batch_size,))

            # now sample from the two dataloaders
            nb_t_minus_samples = (random_sampling < x).int().sum().item()
            nb_t_plus_samples = self.training_cfg.train_batch_size - nb_t_minus_samples
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
                    # so reform them when they needed
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

            # finally shuffle the batch so that seen empirical times are mixed
            batch = batch[torch.randperm(batch.shape[0])]

            # check shape and append
            assert batch.shape == (
                self.training_cfg.train_batch_size,
                *self.data_shape,
            ), f"Expecting sample shape {(self.training_cfg.train_batch_size, *self.data_shape)}, got {batch.shape}"

            yield t, batch

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
            self.training_cfg.train_batch_size,
            *self.data_shape,
        ), f"Expecting batch shape {(self.training_cfg.train_batch_size, *self.data_shape)}, got {batch.shape}"

        # Sample Gaussian noise
        noise = torch.randn_like(batch)

        # Sample a random diffusion timestep
        diff_timesteps = torch.randint(
            0, self.dynamic.config["num_train_timesteps"], (batch.shape[0],), device=self.accelerator.device
        )

        # Forward diffusion process
        noisy_batch = self.dynamic.add_noise(batch, noise, diff_timesteps)  # type: ignore

        # Encode time
        video_time_codes = self.video_time_encoding.forward(time, batch.shape[0])

        # Get model predictions
        pred = self._get_net_pred(noisy_batch, diff_timesteps, video_time_codes)

        # Compute loss
        assert (
            self.dynamic.config["prediction_type"] == "epsilon"
        ), f"Expecting epsilon prediction type, got {self.dynamic.config['prediction_type']}"
        loss = self._loss(pred, noise)

        # Backward pass
        self.accelerator.backward(loss)

        # Gradient clipping
        grad_norm = None
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.net.parameters(),
                self.training_cfg.max_grad_norm,
            )

        # Optimization step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Update global opt step & log the loss
        self.global_optimization_step += 1
        self.accelerator.log(
            {
                "loss": loss.item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "epoch": self.epoch,
                "step": self.global_optimization_step,
                "time": time,
                "L2 gradient norm": grad_norm,
            },
            step=self.global_optimization_step,
        )
        # Wake me up at 3am if loss is NaN
        if torch.isnan(loss) and self.accelerator.is_main_process:
            msg = f"Loss is NaN at epoch {self.global_epoch}, step {self.global_optimization_step}, time {time}"
            wandb.alert(
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            self.logger.critical(msg)
            # TODO: restart from previous checkpoint

    def _get_net_pred(self, noisy_batch: Tensor, diff_timesteps: Tensor | float | int, video_time_codes: Tensor):
        net_type = type(self.accelerator.unwrap_model(self.net))
        if net_type == UNet2DModel:
            return self.net.forward(noisy_batch, diff_timesteps, class_labels=video_time_codes, return_dict=False)[0]  # type: ignore
        elif net_type == UNet2DConditionModel:
            return self.net.forward(
                noisy_batch,
                diff_timesteps,
                encoder_hidden_states=video_time_codes.unsqueeze(1),  # type: ignore
                return_dict=False,
            )[0]
        else:
            raise ValueError(f"Expecting UNet2DModel or UNet2DConditionModel, got {net_type}")

    def _loss(self, pred, target):
        """All the hard work should happen before..."""
        criterion = torch.nn.MSELoss()
        loss = criterion(pred, target)
        return loss

    @torch.inference_mode()
    def _evaluate(
        self,
        dataloaders: dict[float, DataLoader],
        pbar_manager: Manager,
    ):
        """
        Generate inference trajectories, compute metrics and save the model if best to date.

        Should be called by all processes.

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
        inverted_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(self.dynamic.config)  # type: ignore
        inverted_scheduler.set_timesteps(self.training_cfg.eval_nb_diffusion_timesteps)
        # duplicate the scheduler to not mess with the training one
        inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(self.dynamic.config)  # type: ignore
        inference_scheduler.set_timesteps(self.training_cfg.eval_nb_diffusion_timesteps)

        # Misc.
        torch.cuda.empty_cache()
        self.logger.info(f"Starting evaluation on process ({self.accelerator.process_index})", main_process_only=False)

        # At first perform some pure sample generation
        # sample Gaussian noise (always the same one throughout training)
        # TODO: save it once
        rng = torch.Generator(self.accelerator.device).manual_seed(42)
        noise = torch.randn(
            self.training_cfg.eval_batch_size,
            self.accelerator.unwrap_model(self.net).config["in_channels"],
            self.accelerator.unwrap_model(self.net).config["sample_size"],
            self.accelerator.unwrap_model(self.net).config["sample_size"],
            device=self.accelerator.device,
            generator=rng,
        )

        # sample a random video time (always the same one throughout training)
        # TODO: save it once
        random_video_time = torch.rand(self.training_cfg.eval_batch_size, device=self.accelerator.device, generator=rng)
        random_video_time = torch.sort(random_video_time).values  # sort it for better viz
        random_video_time_enc = self.video_time_encoding.forward(random_video_time)

        # generate a sample
        image = noise
        for t in inference_scheduler.timesteps:
            model_output = self._get_net_pred(image, t, random_video_time_enc)
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
        save_eval_artifacts_log_to_wandb(
            image,
            self.saved_artifacts_folder,
            self.global_optimization_step,
            self.accelerator,
            self.logger,
            "simple_generations",
            ["[-1;1] raw", "image min-max", "[-1;1] clipped"],
            captions=[f"time: {t}" for t in random_video_time],
        )

        # Use only 1st dataloader for now TODO
        first_dl = list(dataloaders.values())[0]
        eval_batches_pbar = pbar_manager.counter(
            total=len(first_dl),
            position=2,
            desc="Iterating over evaluation data batches",
            enable=self.accelerator.is_main_process,
            leave=False,
            min_delta=1,
        )
        eval_batches_pbar.refresh()

        # Now generate & evaluate the trajectories
        for batch_idx, batch in enumerate(iter(first_dl)):
            inverted_gauss = batch
            inversion_video_time = self.video_time_encoding.forward(0, batch.shape[0])

            # 1. Save the to-be inverted images
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    batch,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    "starting_samples",
                    ["[-1;1] raw", "image min-max"],
                )

            # 2. Generate the inverted Gaussians
            for t in inverted_scheduler.timesteps:
                model_output = self._get_net_pred(inverted_gauss, t, inversion_video_time)
                inverted_gauss = inverted_scheduler.step(model_output, int(t), inverted_gauss, return_dict=False)[0]
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    inverted_gauss,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    "inversions",
                    ["image min-max"],
                )

            # 3. Regenerate the starting samples from their inversion
            regen = inverted_gauss.clone()
            for t in inference_scheduler.timesteps:
                model_output = self._get_net_pred(regen, t, inversion_video_time)
                regen = inference_scheduler.step(model_output, int(t), regen, return_dict=False)[0]
            if batch_idx == 0:
                save_eval_artifacts_log_to_wandb(
                    regen,
                    self.saved_artifacts_folder,
                    self.global_optimization_step,
                    self.accelerator,
                    self.logger,
                    "regenerations",
                    ["image min-max", "[-1;1] raw", "[-1;1] clipped"],
                )

            # 4. Generate the trajectory from it
            # TODO: parallelize the generation along video time?
            image = inverted_gauss
            video = []
            video_time_pbar = pbar_manager.counter(
                total=self.training_cfg.eval_nb_video_timesteps,
                position=3,
                desc="Generating trajectory: video timesteps ",
                enable=self.accelerator.is_main_process,
                leave=False,
                min_delta=1,
            )

            for video_t_idx, video_time in enumerate(torch.linspace(0, 1, self.training_cfg.eval_nb_video_timesteps)):
                self.logger.debug(f"Video timestep index {video_t_idx} / {self.training_cfg.eval_nb_video_timesteps}")
                video_time_enc = self.video_time_encoding.forward(video_time.item(), batch.shape[0])

                for t in inference_scheduler.timesteps:
                    model_output = self._get_net_pred(image, t, video_time_enc)
                    image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]

                video.append(image)
                video_time_pbar.update()
                # _log_to_file_only(
                #     self.logger,
                #     f"Video time {video_timestep}/{self.training_cfg.eval_nb_video_timesteps}",
                #     INFO,
                # )

            video_time_pbar.close(clear=True)

            video = torch.stack(video)
            # wandb expects (batch_size, video_time, channels, height, width)
            video = video.permute(1, 0, 2, 3, 4)
            expected_video_shape = (
                self.training_cfg.eval_batch_size,
                self.training_cfg.eval_nb_video_timesteps,
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
                    "trajectories",
                    ["image min-max", "video min-max", "[-1;1] raw", "[-1;1] clipped"],
                )

            eval_batches_pbar.update()
            break  # TODO: keep generating on the entire test set and evaluate the trajectories

        # 3. Evaluate the trajectories
        self.logger.warning_once("Should implement evaluation metrics here.")
        # TODO: compute metric on full test data

        eval_batches_pbar.close()
        self.logger.info(f"Finished evaluation on process ({self.accelerator.process_index})", main_process_only=False)

    def _save_model(self):
        """
        Save the net to disk as an independent pretrained model.

        Can be called by all processes (only main will actually save).
        """
        self.accelerator.unwrap_model(self.net).save_pretrained(
            self.model_save_folder, is_main_process=self.accelerator.is_main_process
        )
        self.logger.info(f"Saved model to {self.model_save_folder}")

    def _checkpoint(self):
        """
        Save the current state of the models, optimizers, schedulers, and dataloaders.

        Should be called by all processes as `accelerator.save_state` handles checkpointing
        in DDP setting internally.

        Used to resume training.
        """
        this_chkpt_subfolder = Path(self.checkpointing_cfg.chckpt_save_path) / f"step_{self.global_optimization_step}"
        self.accelerator.save_state(this_chkpt_subfolder.as_posix())

        # Resuming args saved in json file
        training_info_for_resume = {
            "start_instant_batch_idx": self.instant_batch_idx,
            "start_global_optimization_step": self.global_optimization_step,
            "start_global_epoch": self.global_epoch,
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
            checkpoints_list = list(Path(self.checkpointing_cfg.chckpt_save_path).iterdir())
            nb_checkpoints = len(checkpoints_list)
            if nb_checkpoints > self.checkpointing_cfg.checkpoints_total_limit:
                sorted_chkpt_subfolders = sorted(checkpoints_list, key=lambda x: int(x.name.split("_")[1]))
                to_del = sorted_chkpt_subfolders[: -self.checkpointing_cfg.checkpoints_total_limit]
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
) -> ResumingArgs | None:
    """Should be called by all processes"""
    resume_arg = cfg.checkpointing.resume_from_checkpoint
    # 1. first find the correct subfolder to resume from
    chckpt_save_path = Path(cfg.checkpointing.chckpt_save_path)
    if type(resume_arg) == int:
        path = chckpt_save_path / f"step_{resume_arg}"
    else:
        assert resume_arg is True, "Expected resume_from_checkpoint to be True here"
        # Get the most recent checkpoint
        if not chckpt_save_path.exists() and accelerator.is_main_process:
            logger.warning("No 'checkpoints' directory found in run folder; creating one.")
            chckpt_save_path.mkdir()
        accelerator.wait_for_everyone()
        dirs = os.listdir(chckpt_save_path)
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
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


# def _log_to_file_only(logger: MultiProcessAdapter, msg: str, level: int) -> None:
#     """
#     Logs a message to file handlers only.

#     This function iterates over all handlers attached to the logger,
#     and if the handler is a FileHandler, it directly calls its emit
#     method to log the message. This way, only the file handlers will
#     log the message.

#     Args:
#         logger (MultiProcessAdapter): The logger instance with attached handlers.
#         msg (str): The message to log.
#         level (int): The logging level.

#     Returns:
#         None
#     """
#     logger.debug(f"DEBUG: logger.handlers={logger.handlers}")
#     for handler in logger.logger.handlers:
#         if isinstance(handler, FileHandler):  # TODO: doesn't work
#             handler.emit(makeLogRecord({"msg": msg, "level": level}))
#         else:
#             logger.debug(f"Skipping handler of type {type(handler)}")
