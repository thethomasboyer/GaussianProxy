from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch import dtype

from GaussianProxy.conf.training_conf import DataSet, EvaluationStrategy


@dataclass(kw_only=True)
class ProfileConfig:
    enabled: bool
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    with_flops: bool
    export_chrome_trace: bool


@dataclass(kw_only=True)
class InferenceConfig:
    # Choose the experiment (trained model weights)
    root_experiments_path: Path
    project_name: str
    run_name: str

    # Scheduler config
    scheduler_config: Optional[Path]

    # Output directory (where to put the generated images / tensors)
    output_dir: Path

    # Device (ignored if launched with accelerate)
    device: str

    # Optimizations
    compile: bool
    dtype: dtype

    # Data
    dataset: DataSet

    # Evaluations
    evaluation_strategies: list[EvaluationStrategy]

    # Profiling
    profiling: ProfileConfig

    # Temp Dir
    tmpdir_location: str
