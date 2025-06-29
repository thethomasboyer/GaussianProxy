# Copyright 2024 Thomas Boyer. All rights reserved.

################################## launcher.py #################################
# This script acts as a wrapper-launcher.
#
# It performs the following tasks:
# - configure SLURM args (if enabled)
# - copy the code and experiment config to the experiment folder
# (to ensure that is is not modified if the actual job launch is delayed,
# which is quite expected...)
# - show a git diff on the code & config and ask for confirmation
# - configure accelerate
# - set some environment variables
# - submit the task (or run it directly)


import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from logging import Logger
from os import get_terminal_size
from pathlib import Path
from pprint import pformat

import hydra
import submitit
from git import Repo
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.traceback import install
from termcolor import colored

from GaussianProxy.conf.training_conf import Config

# Load the config *object* from the training config file
from my_conf.my_training_conf import config

# Register the `config` *object*
DEFAULT_CONFIG_PATH = "my_conf"
DEFAULT_CONFIG_NAME = "experiment_conf"
cs = ConfigStore.instance()
cs.store(name=DEFAULT_CONFIG_NAME, node=config)

# nice tracebacks
install(width=200)

# wait for this amount of seconds before automatically launching the job when debug flag is set
CONFIRMATION_TIME_LAUNCH_DEBUG_SEC = 5

# how much to wait before resubmitting the task at SIGUSR2 receival at timeout, in seconds
SLEEP_TIME_AFTER_SIGNAL_RECEIVED_SEC = 3 * 60


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: Config) -> None:
    # Hydra config
    hydra_cfg = HydraConfig.get()

    # Logging
    logger: Logger = logging.getLogger(Path(__file__).stem)
    log_file_path = Path(hydra_cfg.run.dir) / "logs.log"
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    try:
        terminal_width = get_terminal_size().columns
    except OSError:
        terminal_width = 80
    logger.info("-" * max(0, terminal_width - 44))
    logger.info("%sLaunching new run", " " * max(0, terminal_width // 2 - 22))

    # SLURM
    if cfg.slurm.enabled:
        # TODO: use the Submitit Launcher plugin (https://hydra.cc/docs/plugins/submitit_launcher)
        executor = submitit.AutoExecutor(
            folder=cfg.slurm.output_folder,
            slurm_max_num_timeout=(min(cfg.slurm.max_num_requeue, 2) if cfg.debug else cfg.slurm.max_num_requeue),
        )

        if cfg.debug is True:
            qos = "dev"
            if cfg.slurm.qos != "qos_gpu-dev":
                logger.warning("debug: setting slurm.qos from %s to 'qos_gpu-dev'", cfg.slurm.qos)
            if cfg.slurm.total_job_time is not None and cfg.slurm.total_job_time > 2 * 60:
                logger.warning("debug: setting slurm.total_job_time from %s to 2 * 60 = 120", cfg.slurm.total_job_time)
                cfg.slurm.total_job_time = 2 * 60
        else:
            qos = cfg.slurm.qos

        additional_parameters = {
            "hint": "nomultithread",
            "mail_user": cfg.slurm.email,
            "mail_type": "FAIL",
        }

        # get the total number of cpus
        match cfg.slurm.constraint:
            case "a100":
                cpus_per_task = int(64 * cfg.slurm.num_gpus / 8)
            case "h100":
                cpus_per_task = int(96 * cfg.slurm.num_gpus / 4)
            case s if isinstance(s, str) and s.startswith("v100"):
                assert cfg.slurm.partition is None, (
                    f"Expected partition to not be set when constraint is set; got {cfg.slurm.partition}"
                )
                cpus_per_task = int(40 * cfg.slurm.num_gpus / 4)
            case None:
                assert cfg.slurm.partition is not None, "Partition must be set when constraint is None"
                match cfg.slurm.partition:
                    case "gpu_p2" | "gpu_p2s" | "gpu_p2l":
                        cpus_per_task = int(24 * cfg.slurm.num_gpus / 8)
                    case _:
                        raise ValueError(f"Unknown partition '{cfg.slurm.partition}'")
            case _:
                raise ValueError(f"Unknown constraint '{cfg.slurm.constraint}'")

        # get correct QOS
        if cfg.slurm.constraint is not None and "v100" in cfg.slurm.constraint:
            slurm_qos = f"qos_gpu-{qos}"
        else:
            slurm_qos = f"qos_gpu_{cfg.slurm.constraint}-{qos}"

        executor.update_parameters(
            slurm_job_name=f"{cfg.project}-{cfg.run_name}",
            slurm_constraint=cfg.slurm.constraint,
            slurm_partition=cfg.slurm.partition,
            slurm_nodes=cfg.slurm.nodes,
            slurm_ntasks_per_node=1,
            slurm_gres=f"gpu:{cfg.slurm.num_gpus}",
            slurm_cpus_per_task=cpus_per_task,
            slurm_additional_parameters=additional_parameters,
            slurm_signal_delay_s=cfg.slurm.send_timeout_signal_n_minutes_before_end * 60,
            slurm_qos=slurm_qos,
            slurm_account=cfg.slurm.account,
        )
        if cfg.slurm.total_job_time is not None:
            executor.update_parameters(slurm_time=cfg.slurm.total_job_time)
        if cfg.slurm.job_launch_delay is not None:
            executor.update_parameters(additional_parameters={"begin": f"now+{cfg.slurm.job_launch_delay}"})

    # CL overrides
    overrides = hydra_cfg.overrides.task
    script_name = cfg.script

    # Create experiment folder & copy config
    # (to prevent config modif when delaying launches)
    task_config_path, launcher_config_name, code_parent_folder = prepare_and_confirm_launch(cfg, hydra_cfg, logger)

    # Task
    task = Task(
        cfg,
        overrides,
        task_config_path,
        launcher_config_name,
        code_parent_folder,
        script_name,
        logger,
    )

    # Submit
    if cfg.slurm.enabled:
        job = executor.submit(task)  # pyright: ignore[ reportPossiblyUnboundVariable]
        logger.info("Task submitted:")
        logger.info(pformat(executor.parameters))  # pyright: ignore[reportPossiblyUnboundVariable]
        logger.info("-" * max(0, terminal_width - 44))
        # Monitor
        if cfg.slurm.monitor:
            submitit.helpers.monitor_jobs([job])  # pyright: ignore[reportArgumentType]
    else:
        task()


class Task:
    """
    Represents an `accelerate launch <script>.py` command call to the system.

    When `__call__`'ed, `Task` will:
    - pass the given config to `<script>.py`
    - configure `accelerate` with the given config
    - set some environment variables
    - submit the command to the system

    Can be called directly or submitted to SLURM with `submitit`.
    """

    def __init__(
        self,
        cfg: Config,
        overrides: list[str],
        task_config_path: Path,
        task_config_name: Path,
        code_and_config_parent_folder: Path,
        script_name: str,
        logger: Logger,
    ):
        self.cfg = cfg
        self.overrides = overrides
        self.task_config_path = task_config_path
        self.task_config_name = task_config_name
        self.code_and_config_parent_folder = code_and_config_parent_folder
        self.script_name = script_name
        self.logger = logger

    def __call__(self):
        # Accelerate config
        accelerate_cfg = ""
        launch_args = self.cfg.accelerate.launch_args

        # handle num_processes or gpu_ids
        if "num_processes" not in launch_args or launch_args.num_processes is None:  # pyright: ignore ReportOperatorIssue
            if self.cfg.slurm.enabled:
                actual_nb_gpus = self.cfg.slurm.num_gpus
            elif launch_args.gpu_ids == "all":
                from torch.cuda import device_count  # import here to avoid slow script starting

                actual_nb_gpus = device_count()
                self.logger.debug(f"Found {actual_nb_gpus} GPU ids from torch.cuda.device_count()")
            else:
                tmp_gpu_ids = launch_args.gpu_ids.split(",")
                actual_nb_gpus = [e.isdigit() for e in tmp_gpu_ids].count(True)
                assert actual_nb_gpus > 0, f"No GPU found in arg `gpu_ids`: {launch_args.gpu_ids}"
                self.logger.debug(f"Found {actual_nb_gpus} GPU ids from passed `gpu_ids` argument")
            launch_args.num_processes = actual_nb_gpus

        for (
            cfg_item_name,
            cfg_item_value,
        ) in launch_args.items():  # pyright: ignore reportAttributeAccessIssue
            if cfg_item_value is True or cfg_item_value in ["True", "true"]:
                accelerate_cfg += f"--{cfg_item_name} "
            elif cfg_item_value is False or cfg_item_value in ["False", "false"]:
                pass
            else:
                accelerate_cfg += f"--{cfg_item_name} {cfg_item_value} "

        # Environment variables
        env_vars = os.environ.copy()

        env_vars["PYTHONPATH"] = f"{self.code_and_config_parent_folder.as_posix()}:{os.environ.get('PYTHONPATH', '')}"

        if self.cfg.debug:
            accelerate_cfg += "--debug "
            env_vars["HYDRA_FULL_ERROR"] = "1"

        if self.cfg.accelerate.offline:
            env_vars["WANDB_MODE"] = "offline"
            env_vars["HF_DATASETS_OFFLINE"] = "1"

        if self.cfg.tmpdir_location is not None:
            Path(self.cfg.tmpdir_location).mkdir(parents=True, exist_ok=True)
            env_vars["TMPDIR"] = self.cfg.tmpdir_location

        # Launched command
        final_cmd = f"accelerate launch {accelerate_cfg} {self.code_and_config_parent_folder.as_posix()}/GaussianProxy/{self.script_name}.py --config-path {self.task_config_path} --config-name {self.task_config_name}"

        for override in self.overrides:
            final_cmd += f" {override}"

        try:
            terminal_width = get_terminal_size().columns
        except OSError:
            terminal_width = 80
        self.logger.info("=" * max(0, terminal_width - 44))
        self.logger.info(f"Executing command: {final_cmd}")
        self.logger.info("=" * max(0, terminal_width - 44))

        # Execute command
        subprocess.run(final_cmd, shell=True, check=True, env=env_vars)

    def checkpoint(self):
        """
        Method called by submitit when the Task times out and `cfg.slurm.max_num_requeue` is not reached.

        It performs 3 actions:
        1. write a flag file indicating the training job to checkpoint and stop training to the specified checkpoint folder
        2. wait a bit
        3. resubmit the job with the same config, but with `resume_from_checkpoint` set to True.
        """
        # 1. Write the flag file
        # get the checkpointing folder like in GaussianProxy/utils/misc.py:create_repo_structure
        this_run_folder = Path(self.cfg.exp_parent_folder, self.cfg.project, self.cfg.run_name)
        if OmegaConf.is_missing(self.cfg.checkpointing, "chckpt_base_path"):
            self.logger.warning(
                f"No checkpointing *base* folder was specified: writing run resume flag file under {this_run_folder}/checkpoints"
            )
            chckpt_save_path = Path(this_run_folder, "checkpoints")
        else:
            chckpt_save_path = Path(self.cfg.checkpointing.chckpt_base_path)
        # write the flag file
        chckpt_save_path.mkdir(parents=True, exist_ok=True)
        chckpt_flag_file = Path(chckpt_save_path, "checkpointing_flag.txt")
        chckpt_flag_file.touch(exist_ok=True)
        self.logger.info(f"Touched checkpointing flag file; sleeping for {SLEEP_TIME_AFTER_SIGNAL_RECEIVED_SEC}")

        # 2. Wait a bit
        time.sleep(SLEEP_TIME_AFTER_SIGNAL_RECEIVED_SEC)

        # 3. Resubmit the job
        # If the run was not set to resume from the latest checkpoint,
        # (ie resume_from_checkpoint=True), then change that argument
        cfg_copy = self.cfg.copy()  # pyright: ignore[reportAttributeAccessIssue]
        if (
            type(cfg_copy.checkpointing.resume_from_checkpoint) is int
            or cfg_copy.checkpointing.resume_from_checkpoint is False
        ):
            cfg_copy.checkpointing.resume_from_checkpoint = True
        # resubmit
        callable_method = Task(
            cfg_copy,
            self.overrides,
            self.task_config_path,
            self.task_config_name,
            self.code_and_config_parent_folder,
            self.script_name,
            logging.getLogger(__name__),
        )
        return submitit.helpers.DelayedSubmission(callable_method)


WRITABLE_FILES_OR_FOLDERS = [
    "logs.log",
    ".hydra",
    "net_summary.txt",
    "run_id.txt",
    "saved_model",
    ".git",
    "train_samples.json",
    "test_samples.json",
    "train_samples.parquet",
    "test_samples.parquet",
    "checkpoints",  # last checkpoint will be immediatly re-written at run resume (TODO: don't)
]


def _set_files_read_only(directory: Path) -> None:
    """
    Recursively set all files in the given directory to read-only,
    excluding files in WRITABLE_FILES_OR_FOLDERS.

    Args:
        directory (Path): The directory containing the files to set to read-only.
    """

    for p in directory.rglob("*"):
        is_writable = (
            any(parent.name in WRITABLE_FILES_OR_FOLDERS for parent in p.parents) or p.name in WRITABLE_FILES_OR_FOLDERS
        )
        if p.is_file() and not is_writable:
            p.chmod(0o444)  # 444 = r--


def _get_config_path_and_name(hydra_cfg: HydraConf) -> tuple[Path, Path]:
    """Returns the 'config_path' and 'config_name' args passed to hydra.

    Arguments
    =========
    - hydra_cfg

        The hydra-specific config automatically retrieved by hydra (typically accessed with `hydra.core.hydra_config.HydraConfig.get()`)

    Returns
    =======
    - config_path: `Path`

        The search path where hydra has searched the config file used to launch the job.

    - config_name: `Path`

        The name of the config file used to launch the job.
    """
    # 1. Get the config_path (passed as config_path to hydra)
    launcher_config_path = None
    for cfg_source in hydra_cfg.runtime.config_sources:
        if cfg_source.provider == "main":
            assert (  # just a quick check
                cfg_source.schema == "file"
            ), "Internal error: Expected 'file://' schema from 'main' config source"
            launcher_config_path = Path(cfg_source.path)
            break
    if launcher_config_path is None:
        raise RuntimeError("Could not find main config path")

    # 2. Get the config name
    cfg_name = hydra_cfg.job.config_name
    if cfg_name is None:
        raise RuntimeError(f"Could not find main config name at hydra_cfg.job.config_name: {cfg_name}")
    task_config_name = Path(cfg_name)

    return launcher_config_path, task_config_name


CODE_FOLDER_NAME = "GaussianProxy"
CONFIG_FOLDER_NAME = "my_conf"
THINGS_TO_COPY = [CODE_FOLDER_NAME, CONFIG_FOLDER_NAME]
# path must be relative to the *launcher* script (this script) parent folder

THINGS_TO_GITIGNORE = [
    "__pycache__",
    "checkpoints",
    "logs.log",
    "saved_model",
    "saved_artifacts",
    ".gitignore",
    ".hydra",
    "net_summary.txt",
    "run_id.txt",
    "*.err",
    "*.out",
    "*.sh",
    "*.pkl",
    "my_conf/my_inference_conf.py",  # do not git diff the user inference config (for now: TODO use it!)
    "train_samples.json",
    "test_samples.json",
]


def _get_specific_diffs(repo: Repo, prev_commit: str, new_commit: str, path: str | None = None) -> str:
    """
    Get git diff for a specific path.

    Args:
        repo: Git repo object
        prev_commit: Previous commit hash
        new_commit: New commit hash
        path: List of path to include in diff (None for all)

    Returns:
        Formatted diff as string
    """
    args = ["--unified=0", "--color", prev_commit, new_commit]
    if path is not None:
        args.extend(["--", path])

    return repo.git.diff(*args)


def _get_code_changes_summary(
    repo: Repo, prev_commit: str, new_commit: str, exclude_path: str | None = None
) -> dict[str, int]:
    """
    Get summary of changed lines per file in code files.

    Args:
        repo: Git repo object
        prev_commit: Previous commit hash
        new_commit: New commit hash
        exclude_path: Path to exclude (typically config path)

    Returns:
        Dict mapping filenames to number of changed lines
    """
    # Get the list of changed files
    changed_files = repo.git.diff("--name-only", prev_commit, new_commit).splitlines()

    # Filter out excluded paths
    if exclude_path is not None:
        changed_files = [f for f in changed_files if not any(f.startswith(path) for path in exclude_path)]

    # Get stat for each file
    result = {}
    for file in changed_files:
        try:
            # Get number of insertions and deletions
            stat = repo.git.diff("--numstat", prev_commit, new_commit, "--", file).split()
            if len(stat) >= 2:
                insertions = int(stat[0]) if stat[0] != "-" else 0
                deletions = int(stat[1]) if stat[1] != "-" else 0
                result[file] = insertions + deletions
        except Exception:
            # If we can't get stats for some reason, just mark the file as changed
            result[file] = 1

    return result


def _format_code_changes_summary(summary: dict[str, int]) -> str:
    """Format a summary of code changes."""
    if not summary:
        return "No code changes detected."

    # Count total changes
    total_changes = sum(summary.values())
    total_files = len(summary)

    lines = [
        f"Code changes summary: {total_changes} line{'s' if total_changes != 1 else ''} changed in {total_files} file{'s' if total_files != 1 else ''}"
    ]
    for file, count in summary.items():
        lines.append(f"  - {file}: {count} line{'s' if count != 1 else ''} changed")

    return "\n".join(lines)


def prepare_and_confirm_launch(cfg: Config, hydra_cfg: HydraConf, logger: Logger) -> tuple[Path, Path, Path]:
    """
    Copy the task config & code to the experiment folder so that delayed launches do not risk being modified.

    Only files or folders in CODE_TO_COPY are actually copied in the code folder.
    """
    # 1. Get this launcher's config
    launcher_config_path, launcher_config_name = _get_config_path_and_name(hydra_cfg)

    # 2. Get the experiment folder
    hydra_cfg = HydraConfig.get()
    this_experiment_folder = Path(hydra_cfg.run.dir)
    this_experiment_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying code and config to {this_experiment_folder}")

    # 3. Get git hash of current state of affairs
    repo = Repo.init(this_experiment_folder)
    gitignore_path = Path(this_experiment_folder, ".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path.as_posix(), "x") as f:
            for thing in THINGS_TO_GITIGNORE:
                f.write(f"{thing}\n")
    try:
        prev_commit = repo.head.commit
    except ValueError:
        prev_commit = None

    # 4. Copy the config to the experiment folder
    dst_config_path = Path(this_experiment_folder, launcher_config_path.name)
    if dst_config_path.exists():
        shutil.rmtree(dst_config_path)
    task_config_path = shutil.copytree(
        launcher_config_path,
        dst_config_path,
        dirs_exist_ok=True,
    )
    _set_files_read_only(dst_config_path)
    logger.debug(f"Copied config {launcher_config_name} to {this_experiment_folder}")

    # 5. Copy the code to the experiment folder
    for file_or_folder_name in THINGS_TO_COPY:
        source_path = Path(cfg.launcher_script_parent_folder, file_or_folder_name)
        destination_path = Path(this_experiment_folder, file_or_folder_name)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        p = Path(file_or_folder_name)
        if p.is_file():
            if destination_path.exists():
                destination_path.unlink()
            shutil.copy2(source_path, destination_path)
        else:
            if destination_path.exists():
                shutil.rmtree(destination_path)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    _set_files_read_only(this_experiment_folder)
    logger.debug(f"Copied files or folders: {THINGS_TO_COPY} to {this_experiment_folder}")

    # 6. Get git hash of new state of affairs
    for f in THINGS_TO_COPY:
        # we need to add everything in case the repo was not initialized yet
        repo.git.add(f)
    if repo.is_dirty():
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        repo.git.commit("-m", f"automated commit of dirty repo at launch time: {current_time}")
        logger.debug(f"Committed changes at {this_experiment_folder} with hash {repo.head.commit.hexsha}")

    # first instanciation of this repo
    if prev_commit is None:
        logger.info(
            f"First instantiation of {this_experiment_folder}: not showing (inexistent) git diff and proceeding with run launch"
        )
        return task_config_path, launcher_config_name, this_experiment_folder

    # otherwise show diff and ask for confirmation
    new_commit = repo.head.commit

    diff_mode = cfg.diff_mode

    match diff_mode:
        case "full":
            # Show full diff of everything
            full_diff = _get_specific_diffs(repo, prev_commit.hexsha, new_commit.hexsha)
            logger.info(f"Git diff with previous commit: {prev_commit.message}\n{full_diff}")
        case "config_only":
            # Show only config diff + code summary
            logger.warning("Diff mode set to 'config_only': showing only config diff and code changes summary")
            config_diff = _get_specific_diffs(repo, prev_commit.hexsha, new_commit.hexsha, CONFIG_FOLDER_NAME)
            code_changes = _get_code_changes_summary(repo, prev_commit.hexsha, new_commit.hexsha, CONFIG_FOLDER_NAME)
            if config_diff == "":
                logger.info(f"No config changes detected with previous commit: {prev_commit.message}")
            else:
                logger.info(f"Git diff for config files with previous commit: {prev_commit.message}:\n{config_diff}")
            logger.info(_format_code_changes_summary(code_changes))
        case _:
            raise ValueError(f"Unknown diff mode: {diff_mode}")

    # 7. Get confirmation of launch, otherwise revert changes
    # first get confirmation
    if cfg.debug:
        try:
            logger.info(
                f"Proceeding with launch for debug run {bold(this_experiment_folder.name)} in {CONFIRMATION_TIME_LAUNCH_DEBUG_SEC} seconds"
            )
            time.sleep(CONFIRMATION_TIME_LAUNCH_DEBUG_SEC)
            do_launch = True
        except KeyboardInterrupt:
            do_launch = False
    else:
        try:
            # Add option to show full diff if not already shown
            if diff_mode != "full":
                confirmation = input("Show code diff additionally? (y/[n]): ")
                if confirmation.lower() == "y":
                    code_diff = _get_specific_diffs(repo, prev_commit.hexsha, new_commit.hexsha, CODE_FOLDER_NAME)
                    logger.info(f"Code diff:\n{code_diff}")

            confirmation = input(f"Proceed with launch for run {bold(this_experiment_folder.name)}? (y/[n]): ")
            if confirmation != "y":
                do_launch = False
            else:
                do_launch = True
        except KeyboardInterrupt:
            do_launch = False

    # if not, revert changes
    if not do_launch:
        logger.warning("Launch aborted")
        if prev_commit is not None:
            logger.warning(f"Reverting changes in the experiment folder {this_experiment_folder}")
            repo.git.reset("--hard", prev_commit)
        else:
            repo.git.update_ref("-d", "HEAD")
            logger.warning(
                f"Deleting HEAD ref in the experiment folder {this_experiment_folder} as it was the first commit"
            )
        sys.exit()
    # otherwise, proceed with launch
    return task_config_path, launcher_config_name, this_experiment_folder


# Redefine bold from utils.misc here so that long imports at utils.misc top level
# (like torch...) do not slow down launcher.py
def bold(s: str | int) -> str:
    return colored(s, None, None, ["bold"])


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
