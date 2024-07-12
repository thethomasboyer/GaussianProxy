# Imports
from math import ceil
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from enlighten import Manager, get_manager
from numpy import ndarray
from PIL import Image
from rich.traceback import install
from torch import IntTensor, Tensor

from conf.inference_conf import InferenceConfig
from conf.training_conf import ForwardNoising, InvertedRegeneration, SimpleGeneration
from my_conf.my_inference_conf import inference_conf
from utils.data import NumpyDataset
from utils.misc import _normalize_elements_for_logging
from utils.models import VideoTimeEncoding

# No grads
torch.set_grad_enabled(False)

# Nice tracebacks
install()


def main(cfg: InferenceConfig) -> None:
    ###############################################################################################
    # Check paths
    ###############################################################################################
    project_path = cfg.root_experiments_path / cfg.project_name
    assert project_path.exists(), f"Project path {project_path} does not exist."

    run_path = project_path / cfg.run_name
    assert run_path.exists(), f"Run path {run_path} does not exist."
    print("run path:", run_path)

    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    print("output dir:", cfg.output_dir)

    ###############################################################################################
    # Load Model
    ###############################################################################################
    # denoiser
    net: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(run_path / "saved_model" / "net")  # type: ignore
    net.to(cfg.device)
    net.eval()

    # time encoder
    video_time_encoder: VideoTimeEncoding = VideoTimeEncoding.from_pretrained(  # type: ignore
        run_path / "saved_model" / "video_time_encoder"
    )
    video_time_encoder.to(cfg.device)
    video_time_encoder.eval()

    # dynamic
    dynamic: DDIMScheduler = DDIMScheduler.from_pretrained(run_path / "saved_model" / "dynamic")

    ###############################################################################################
    # Load Starting Images
    ###############################################################################################
    database_path = Path(cfg.dataset.path)
    subdirs = sorted([e for e in database_path.iterdir() if e.is_dir() and not e.name.startswith(".")])

    # we are only interested in the first subdir: timestep 0
    assert subdirs[0].name == "1", f"Expected '1' as first subdir, got {subdirs[0].name} in subdir list: {subdirs}"

    # I reuse the NumpyDataset class from the training script to load the images consistently
    starting_samples = list(subdirs[0].glob("*.npy"))
    starting_ds = NumpyDataset(starting_samples, cfg.dataset.transforms, cfg.expected_initial_data_range)
    print("Built dataset:")
    print(starting_ds)

    # now select batch_size samples
    sample_idxes: list[int] = (
        np.random.default_rng().choice(len(starting_ds), cfg.nb_generated_samples, replace=False).tolist()
    )
    tensors = starting_ds.__getitems__(sample_idxes)
    starting_batch = torch.stack(tensors).to(cfg.device)
    print(f"Selected {len(sample_idxes)} samples to run inference from.")

    ###############################################################################################
    # Inference passes
    ###############################################################################################
    pbar_manager: Manager = get_manager()  # type: ignore

    for eval_strat_idx, eval_strat in enumerate(cfg.evaluation_strategies):
        print(f"Running evaluation strategy {eval_strat_idx+1}/{len(cfg.evaluation_strategies)}:\n{eval_strat}")
        if type(eval_strat) is SimpleGeneration:
            raise NotImplementedError
        elif type(eval_strat) is ForwardNoising:
            forward_noising(cfg, eval_strat, net, video_time_encoder, dynamic, starting_batch, pbar_manager)
        elif type(eval_strat) is InvertedRegeneration:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown evaluation strategy {eval_strat}")


def forward_noising(
    cfg: InferenceConfig,
    eval_strat: ForwardNoising,
    net: UNet2DConditionModel,
    video_time_encoder: VideoTimeEncoding,
    dynamic: DDIMScheduler,
    batch: Tensor,
    pbar_manager: Manager,
):
    # 0. Setup scheduler
    inference_scheduler: DDIMScheduler = DDIMScheduler.from_config(dynamic.config)  # type: ignore
    inference_scheduler.set_timesteps(eval_strat.nb_diffusion_timesteps)

    # 1. Save the to-be noised images
    save_images_or_videos(
        batch,
        cfg.output_dir,
        eval_strat.name,
        "starting_samples",
        ["[-1;1] raw", "image min-max"],
    )

    # 2. Sample Gaussian noise and noise the images until some step
    noise = torch.randn(batch.shape, dtype=batch.dtype, device=batch.device)
    noise_timestep_idx = int((1 - eval_strat.forward_noising_frac) * len(inference_scheduler.timesteps))
    noise_timestep = inference_scheduler.timesteps[noise_timestep_idx].item()
    noise_timesteps: IntTensor = torch.full(  # type: ignore
        (batch.shape[0],),
        noise_timestep,
        device=batch.device,
        dtype=torch.int64,
    )
    slightly_noised_sample = inference_scheduler.add_noise(batch, noise, noise_timesteps)
    save_images_or_videos(
        slightly_noised_sample,
        cfg.output_dir,
        eval_strat.name,
        "noised_samples",
        ["image min-max", "[-1;1] raw", "[-1;1] clipped"],
    )

    # 3. Generate the trajectory from it
    # the generation is parallelized along video time, but this
    # usefull if small inference batch size only
    video = []
    nb_vid_batches = ceil(cfg.nb_video_timesteps / cfg.nb_video_times_in_parallel)

    video_time_pbar = pbar_manager.counter(
        total=nb_vid_batches,
        position=1,
        desc="Video timesteps batches",
        leave=False,
    )
    video_time_pbar.refresh()

    video_times = torch.linspace(0, 1, cfg.nb_video_timesteps, device=cfg.device)

    for vid_batch_idx in range(nb_vid_batches):
        diff_time_pbar = pbar_manager.counter(
            total=len(inference_scheduler.timesteps),
            position=2,
            desc="Diffusion timesteps" + " " * 4,
            leave=False,
            count=noise_timestep_idx,
        )
        diff_time_pbar.refresh()

        start = vid_batch_idx * cfg.nb_video_times_in_parallel
        end = (vid_batch_idx + 1) * cfg.nb_video_times_in_parallel
        video_time_batch = video_times[start:end]
        # at this point video_time_batch is at most cfg.nb_video_times_in_parallel long;
        # we need to duplicate the video_time_encoding to match the batch size!
        video_time_enc = video_time_encoder.forward(video_time_batch)
        # of course it's better to do it now than before the video_time_encoder.forward :)
        # (although perf win is probably negligible)
        video_time_enc = video_time_enc.repeat_interleave(cfg.nb_generated_samples, dim=0)

        image = torch.cat([slightly_noised_sample.clone() for _ in range(len(video_time_batch))])
        # shape: (len(video_time_batch)*batch_size, channels, height, width),
        # where len(video_time_batch) = cfg.nb_video_times_in_parallel for at least all but the last batch

        for t in inference_scheduler.timesteps[noise_timestep_idx:]:
            model_output: Tensor = net.forward(
                image, t, encoder_hidden_states=video_time_enc.unsqueeze(1), return_dict=False
            )[0]
            image = inference_scheduler.step(model_output, int(t), image, return_dict=False)[0]
            diff_time_pbar.update()

        diff_time_pbar.close()
        video.append(image)
        video_time_pbar.update()

    video_time_pbar.close()

    # video is a list of ceil(nb_video_timesteps / nb_video_times_in_parallel) elements, each of shape
    # (len(video_time_batch) * batch_size, channels, hight, width)
    video = torch.cat(video)  # (nb_video_timesteps * nb_generated_samples, channels, height, width)
    video = video.split(cfg.nb_generated_samples)
    video = torch.stack(video)
    # save_images_or_videos expects (video_time, batch_size, channels, height, width)
    expected_video_shape = (
        cfg.nb_video_timesteps,
        cfg.nb_generated_samples,
        net.config["out_channels"],
        net.config["sample_size"],
        net.config["sample_size"],
    )
    assert video.shape == expected_video_shape, f"Expected video shape {expected_video_shape}, got {video.shape}"
    save_images_or_videos(
        video,
        cfg.output_dir,
        eval_strat.name,
        "trajectories",
        ["image min-max", "video min-max", "[-1;1] raw", "[-1;1] clipped"],
    )


def save_images_or_videos(
    tensor: Tensor, output_dir: Path, eval_strat_name: str, artifact_name: str, norm_methods: list[str]
):
    base_save_path = output_dir / eval_strat_name
    base_save_path.mkdir(exist_ok=True, parents=True)

    # Save some raw images / trajectories to disk
    file_path = base_save_path / f"{artifact_name}.pt"
    torch.save(tensor, file_path)
    print(f"    Saved raw {artifact_name} of shape {tensor.shape} to {file_path.name}")

    normalized_elements = _normalize_elements_for_logging(tensor, norm_methods)

    match tensor.ndim:
        case 5:  # videos
            for norm_method, normed_vids in normalized_elements.items():
                # save the videos in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.mp4"
                save_grid_of_videos(normed_vids, save_path)
                print(f"    Saved {norm_method} normalized {artifact_name} to {save_path}")
        case 4:  # images
            for norm_method, normed_imgs in normalized_elements.items():
                # save the images in a grid
                save_path = base_save_path / f"{artifact_name}_{norm_method}.png"
                save_grid_of_images(normed_imgs, save_path)
                print(f"    Saved {norm_method} normalized {artifact_name} to {save_path}")


def save_grid_of_videos(videos_tensor: ndarray, save_path: Path, grid_rows: int = 4, grid_cols: int = 4):
    # Checks
    assert videos_tensor.ndim == 5, f"Expected 5D tensor, got {videos_tensor.shape}"
    assert videos_tensor.dtype == np.uint8, f"Expected dtype uint8, got {videos_tensor.dtype}"
    assert (
        videos_tensor.min() >= 0 and videos_tensor.max() <= 255
    ), f"Expected [0;255] range, got [{videos_tensor.min()}, {videos_tensor.max()}]"

    # Convert tensor to a grid of videos
    fps = max(1, int(len(videos_tensor) / 10))
    writer = imageio.get_writer(save_path, mode="I", fps=fps)

    for frame in videos_tensor:
        grid_img = torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4)
        pil_img = grid_img.numpy().transpose(1, 2, 0)
        writer.append_data(pil_img)

    writer.close()


def save_grid_of_images(images_tensor: ndarray, save_path: Path):
    # Checks
    assert images_tensor.ndim == 4, f"Expected 4D tensor, got {images_tensor.shape}"
    assert images_tensor.dtype == np.uint8, f"Expected dtype uint8, got {images_tensor.dtype}"
    assert (
        images_tensor.min() >= 0 and images_tensor.max() <= 255
    ), f"Expected [0;255] range, got [{images_tensor.min()}, {images_tensor.max()}]"

    # Convert tensor to a grid of images
    grid_img = torchvision.utils.make_grid(torch.from_numpy(images_tensor), nrow=4)

    # Convert to PIL Image
    pil_img = Image.fromarray(grid_img.numpy().transpose(1, 2, 0))
    pil_img.save(save_path)


if __name__ == "__main__":
    main(inference_conf)
