from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class Slurm:
    enabled: bool
    monitor: bool
    signal_time: int  # in minutes
    email: str
    output_folder: str
    num_gpus: int
    qos: str
    constraint: Optional[str]
    nodes: int
    account: str
    max_num_requeue: int
    partition: Optional[str]

    def __post_init__(self):
        valid_qos_values = ["dev", "t3", "t4"]
        if self.qos not in valid_qos_values:
            raise ValueError(f"Invalid qos value. Expected one of {valid_qos_values}, got {self.qos}")


@dataclass
class AccelerateLaunchArgs:
    machine_rank: int
    num_machines: int
    rdzv_backend: str
    same_network: str
    mixed_precision: str
    num_processes: Optional[int]
    main_process_port: int
    dynamo_backend: str = "no"
    gpu_ids: str = "all"
    multi_gpu: bool = field(init=False)

    def __post_init__(self):
        cond1 = self.num_processes is not None and self.num_processes > 1
        cond2 = len([gpu_id for gpu_id in self.gpu_ids.split(",") if gpu_id]) > 1
        if cond1 or cond2:
            self.multi_gpu = True
        else:
            self.multi_gpu = False


@dataclass
class Accelerate:
    launch_args: AccelerateLaunchArgs
    offline: bool


@dataclass
class DataSet:
    selected_dists: list[int]
    # data_shape should be tuple[int, int, int] | tuple[int, int], but unions of containers
    # are not yet supported by OmegaConf: https://github.com/omry/omegaconf/issues/144
    data_shape: tuple[int, ...]
    path: str | Path
    transforms: Any
    name: str
    expected_initial_data_range: tuple[float, float] | None


@dataclass
class DataLoader:
    num_workers: Optional[int]
    train_prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool


@dataclass
class Training:
    gradient_accumulation_steps: int
    train_batch_size: int
    max_grad_norm: int
    nb_epochs: int
    nb_time_samplings: int


@dataclass
class EvaluationStrategy:
    nb_diffusion_timesteps: int
    # we must repeat ourselves because hydra actually passes DictConfig objects instead of plain classes...
    name: str = field(default=MISSING)


@dataclass
class InvertedRegeneration(EvaluationStrategy):
    name: str = field(default="InvertedRegeneration")


@dataclass
class SimpleGeneration(EvaluationStrategy):
    name: str = field(default="SimpleGeneration")


@dataclass(kw_only=True)
class ForwardNoising(EvaluationStrategy):
    forward_noising_frac: float
    name: str = field(default="ForwardNoising")


@dataclass
class Evaluation:
    every_n_epochs: int | None
    every_n_opt_steps: int | None
    batch_size: int
    nb_video_timesteps: int
    # the naming convention for the strategy variable names is lowercase + underscore
    # this has to be respected for debug args modification
    strategies: list[EvaluationStrategy]


@dataclass
class Checkpointing:
    checkpoints_total_limit: int
    resume_from_checkpoint: bool | int
    checkpoint_every_n_steps: int
    chckpt_save_path: Path | str


@dataclass
class DDIMSchedulerConfig:
    num_train_timesteps: int
    prediction_type: str  # 'epsilon' or 'v_prediction'
    # clipping
    clip_sample: bool
    clip_sample_range: float
    # dynamic thresholding
    thresholding: bool
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1
    # noise scheduler
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    timestep_spacing: str = "trailing"
    rescale_betas_zero_snr: bool = True


@dataclass
class UNet2DModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    act_fn: str
    class_embed_type: Optional[str]
    center_input_sample: bool = False
    time_embedding_type: str = "positional"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    mid_block_scale_factor: float = 1
    downsample_padding: int = 1
    dropout: float = 0.0
    attention_head_dim: int = 8
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    resnet_time_scale_shift: str = "default"


@dataclass
class UNet2DConditionModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    act_fn: str
    cross_attention_dim: int


NetConfig = UNet2DModelConfig | UNet2DConditionModelConfig


@dataclass
class TimeEncoderConfig:
    encoding_dim: int
    time_embed_dim: int
    flip_sin_to_cos: bool
    downscale_freq_shift: float


@dataclass(kw_only=True)
class Config:
    # Defaults
    defaults: list[str]

    # Model
    dynamic: DDIMSchedulerConfig
    net: Any  # Unions of containers are not supported.......
    time_encoder: TimeEncoderConfig

    # Script
    path_to_script_parent_folder: str
    script: str

    # ExperimentVariables
    exp_parent_folder: str
    project: str
    run_name: str

    # Hydra
    hydra: Any

    # Slurm
    slurm: Slurm

    # Accelerate
    accelerate: Accelerate

    # Miscellaneous
    debug: bool
    profile: bool

    # Caches
    tmpdir_location: Optional[str] = None

    # Experiment tracker
    logger: str
    entity: str

    # Checkpointing
    checkpointing: Checkpointing

    # Dataset
    dataset: DataSet

    # Dataloader
    dataloaders: DataLoader

    # Training
    training: Training

    # Evaluation
    evaluation: Evaluation

    # Optimization
    learning_rate: float
