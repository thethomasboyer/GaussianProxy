import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING


def custom_showwarning(message, category, filename, lineno, _file=None, _line=None):
    print(f"{filename}:{lineno}: {category.__name__}: {message}")


warnings.showwarning = custom_showwarning
warnings.simplefilter("once")


@dataclass(kw_only=True)
class Slurm:
    enabled: bool
    monitor: bool = False
    total_job_time: int | None = None  # in minutes
    send_timeout_signal_n_minutes_before_end: int = 5  # in minutes
    email: str
    output_folder: str
    num_gpus: int
    qos: str
    constraint: str | None
    nodes: int
    account: str
    max_num_requeue: int
    partition: str | None
    job_launch_delay: str | None = None

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
    num_processes: int | None
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
    offline: bool  # TODO: move this arg that does not belong here (make it general like debug)


@dataclass
class DatasetParams:  # TODO: fusion with DataSet
    """
    - `file_extension`: the extension of the files to load, without the dot
    - `key_transform`: a function to transform the subdir name into a timestep
    - `sorting_func`: a function to sort the subdirs
    - `dataset_class`: the class of the dataset to instantiate
    """

    file_extension: str
    key_transform: Any  # should be Callable[[str], int] | Callable[[str], str] ...
    sorting_func: Any  # should be Callable[something...]
    dataset_class: type


@dataclass(kw_only=True)
class DataSet:
    # data_shape should be tuple[int, int, int] | tuple[int, int], but unions of containers
    # are not yet supported by OmegaConf: https://github.com/omry/omegaconf/issues/144
    data_shape: tuple[int, ...]
    path: str | Path = field(default=MISSING)
    transforms: Any
    name: str
    expected_initial_data_range: tuple[float, float] | None
    # same goes for selected_dists: should be list[int] | list[str]...
    selected_dists: list | None = None
    dataset_params: DatasetParams | None = None
    fully_ordered: bool = False
    path_to_single_parquet: str | None = None


@dataclass
class DataLoader:
    num_workers: int | None
    train_prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool


@dataclass
class Training:
    gradient_accumulation_steps: int
    train_batch_size: int
    max_grad_norm: int
    nb_time_samplings: int
    unpaired_data: bool
    as_many_samples_as_unpaired: bool = False


@dataclass(kw_only=True)
class EvaluationStrategy:
    nb_diffusion_timesteps: int
    # we must repeat ourselves because hydra actually passes DictConfig objects instead of plain classes...
    name: str = field(default=MISSING)


@dataclass(kw_only=True)
class InversionRegenerationOnly(EvaluationStrategy):
    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    n_rows_displayed: int


@dataclass(kw_only=True)
class InvertedRegeneration(EvaluationStrategy):
    name: str = field(default="InvertedRegeneration")
    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    nb_video_times_in_parallel: int
    nb_video_timesteps: int
    n_rows_displayed: int
    nb_inversion_diffusion_timesteps: int | None = None


@dataclass(kw_only=True)
class IterativeInvertedRegeneration(InvertedRegeneration):
    name: str = field(default="IterativeInvertedRegeneration")


@dataclass(kw_only=True)
class SimpleGeneration(EvaluationStrategy):
    name: str = field(default="SimpleGeneration")
    plate_name_to_simulate: str | None = None
    n_rows_displayed: int
    nb_generated_samples: int


@dataclass(kw_only=True)
class SimilarityWithTrainData(EvaluationStrategy):
    nb_generated_samples: int
    batch_size: int
    nb_batches_shown: int
    metrics: Any = "cosine"  # should be list[Literal["cosine", "L2"]]
    name: str = field(default="SimilarityWithTrainData")
    n_rows_displayed: int


@dataclass(kw_only=True)
class ForwardNoising(EvaluationStrategy):
    forward_noising_frac: float
    name: str = field(default="ForwardNoising")
    nb_generated_samples: int
    plate_name_to_simulate: str | None = None
    nb_video_times_in_parallel: int
    nb_video_timesteps: int
    n_rows_displayed: int


@dataclass(kw_only=True)
class ForwardNoisingLinearScaling(ForwardNoising):
    forward_noising_frac_start: float
    forward_noising_frac_end: float
    name: str = field(default="ForwardNoisingLinearScaling")


@dataclass(kw_only=True)
class MetricsComputation(EvaluationStrategy):
    nb_samples_to_gen_per_time: int | str
    batch_size: int
    name: str = field(default="MetricsComputation")
    regen_images: bool = True
    nb_recompute: int = 1  # TODO: not used
    selected_times: list
    dtype: str = "float32"
    augmentations_for_metrics_comp: list[str]


@dataclass
class Evaluation:
    every_n_opt_steps: int | None
    batch_size: int
    nb_video_timesteps: int
    # the naming convention for the strategy variable names is lowercase + underscore
    # this has to be respected for debug args modification
    strategies: list[EvaluationStrategy]


@dataclass(kw_only=True)
class Checkpointing:
    checkpoints_total_limit: int
    resume_from_checkpoint: bool | int | str
    checkpoint_every_n_steps: int
    chckpt_base_path: Path


@dataclass(kw_only=True)
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
    timestep_spacing: str
    rescale_betas_zero_snr: bool


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
    class_embed_type: str | None
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


@dataclass(kw_only=True)
class UNet2DConditionModelConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    block_out_channels: tuple[int, ...]
    norm_num_groups: int = 32
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
    defaults: list

    # Model
    dynamic: DDIMSchedulerConfig
    net: Any  # Unions of containers are not supported.......
    time_encoder: TimeEncoderConfig

    # Script
    launcher_script_parent_folder: str
    script: str

    # Experiment Variables
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
    profile: bool = False
    resume_method: str = "rewind"
    diff_mode: str = "config_only"  # "config_only" or "full"

    # Caches
    tmpdir_location: str | None = None

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

    def __post_init__(self):
        """Checks"""
        # dataset
        if not isinstance(self.dataset, DataSet):
            warnings.warn(
                f"Cannot check dataset config because it is not instantiated yet (is {type(self.dataset)})",
                RuntimeWarning,
            )
        else:
            for eval_strat in self.evaluation.strategies:
                if isinstance(eval_strat, MetricsComputation):
                    if self.dataset.selected_dists is not None:
                        if not set(eval_strat.selected_times).issubset(set(self.dataset.selected_dists)):
                            raise ValueError(
                                f"MetricsComputation selected_times {eval_strat.selected_times} not in dataset"
                            )
                    else:
                        warnings.warn(
                            f"Cannot check if MetricsComputation's selected_times {eval_strat.selected_times} are in dataset {self.dataset.name} because no selected_dists were provided",
                            RuntimeWarning,
                        )
        # evaluations
        if any(isinstance(eval_strat, SimilarityWithTrainData) for eval_strat in self.evaluation.strategies):
            sim_strat_index = next(
                index
                for index, strategy in enumerate(self.evaluation.strategies)
                if isinstance(strategy, SimilarityWithTrainData)
            )
            try:
                metrics_comp_index = next(
                    index
                    for index, strategy in enumerate(self.evaluation.strategies)
                    if isinstance(strategy, MetricsComputation)
                )
            except StopIteration as e:
                raise ValueError(
                    f"SimilarityWithTrainData strategy requires MetricsComputation strategy to be present in evaluation strategies, got: {[s.name for s in self.evaluation.strategies]}"
                ) from e
            if sim_strat_index < metrics_comp_index:
                raise ValueError(
                    "SimilarityWithTrainData strategy must come after MetricsComputation strategy in evaluation strategies"
                )
