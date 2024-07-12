from dataclasses import dataclass
from pathlib import Path

from .training_conf import DataSet, EvaluationStrategy


@dataclass(kw_only=True)
class InferenceConfig:
    # Choose the experiment (trained model weights)
    root_experiments_path: Path
    project_name: str
    run_name: str

    # Output directory (where to put the generated images / tensors)
    output_dir: Path

    # Device
    device: str

    # Data
    dataset: DataSet
    expected_initial_data_range: tuple[float, float]

    # Evaluations
    nb_generated_samples: int
    nb_video_times_in_parallel: int
    nb_video_timesteps: int
    evaluation_strategies: list[EvaluationStrategy]
