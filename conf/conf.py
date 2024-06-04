from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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
class TorchCompile:
    do_compile: bool
    fullgraph: bool
    dynamic: bool | None
    backend: str
    mode: str


@dataclass
class DataSet:
    selected_dists: list[int]
    # data_shape should be tuple[int, int, int] | tuple[int, int], but unions of containers
    # are not yet supported by OmegaConf: https://github.com/omry/omegaconf/issues/144
    data_shape: tuple[int, ...]
    path: str | Path
    transforms: Any
    empirical_dists_centers: list[list[int]]
    empirical_dists_vars: list[float]
    nb_samples_per_time: int
    test_split_frac: float
    name: str


@dataclass
class DataLoader:
    num_workers: Optional[int]
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool


@dataclass
class Training:
    gradient_accumulation_steps: int
    train_batch_size: int
    inference_batch_size: int
    max_grad_norm: int
    nb_epochs: int
    nb_time_samplings: int
    eval_every_n_epochs: int | None
    eval_every_n_opt_steps: int | None


@dataclass
class Checkpointing:
    checkpoints_total_limit: int
    resume_from_checkpoint: bool | int
    checkpoint_every_n_steps: int
    chckpt_save_path: Path | str


@dataclass
class DDIMSchedulerConfig:
    num_train_timesteps: int
    clip_sample: bool
    clip_sample_range: float
    prediction_type: str
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
    class_embed_type: str
    center_input_sample: bool = False
    time_embedding_type: str = "positional"
    freq_shift: int = 0
    flip_sin_to_cos: bool = True
    mid_block_scale_factor: float = 1
    downsample_padding: int = 1
    dropout: float = 0.0
    attention_head_dim: int = 8
    norm_num_groups: int = 32
    attn_norm_num_groups: Optional[int] = None
    norm_eps: float = 1e-5
    resnet_time_scale_shift: str = "default"


@dataclass
class Config:
    # Defaults
    defaults: list[str]

    dynamic: DDIMSchedulerConfig
    net: UNet2DModelConfig

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

    # TorchCompile
    torch_compile: TorchCompile

    # Metrics
    compute_fid: bool
    compute_isc: bool
    compute_kid: bool
    kid_subset_size: int
    eval_every_n_epochs: Optional[int]
    eval_every_n_BI_stages: Optional[int]

    # Miscellaneous
    debug: bool
    profile: bool
    tmp_dataloader_path: Optional[Path | str]

    # Logging
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

    # Optimization
    learning_rate: float
    adam_use_fused: bool
