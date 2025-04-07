import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from PIL import Image
from rich.traceback import install
from tqdm.rich import tqdm

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
from GaussianProxy.utils.misc import generate_all_augs

install()

###################################################### Arguments ######################################################
DATASET_BASE_PATH = Path("/projects/static2dynamic/datasets/DiabeticRetinopathy/prepared_dataset/train")
EXTENSION = "jpeg"
DEBUG = False
TRANSFORMS = ["RandomHorizontalFlip"]

######################################################### Info ########################################################
print(f"Augmenting base dataset located at {DATASET_BASE_PATH}", flush=True)
print(f"Using the following transforms: {TRANSFORMS}", flush=True)
print("DEBUG:", DEBUG, flush=True)
sleep(3)


def ending(path: Path, n: int):
    return Path(*path.parts[-n:])


def augment_save_one_file(base_file: Path, aug_subdir_path: Path):
    img = Image.open(base_file)
    augs = generate_all_augs(img, TRANSFORMS)
    for i, aug in enumerate(augs):
        this_aug_save_path = aug_subdir_path / base_file.parent.name / f"{base_file.stem}_aug_{i}.{EXTENSION}"
        if not DEBUG:
            aug.save(this_aug_save_path)
        else:
            print(f"Would have saved augmentation {i} of {ending(base_file, 3)} to {ending(this_aug_save_path, 3)}")


if __name__ == "__main__":
    # get names of subdirs / timestamps
    subdirs_names = [x.name for x in DATASET_BASE_PATH.iterdir() if x.is_dir()]
    print(f"Found {len(subdirs_names)} subdirectories: {subdirs_names}")

    # get all base files to augment
    all_files = list(DATASET_BASE_PATH.glob(f"**/*.{EXTENSION}", recurse_symlinks=True))
    print(f"Found {len(all_files)} base files in total")

    # create augmented subdirs in a adjacent dir to the base dataset one
    aug_subdir_path = DATASET_BASE_PATH.with_name(DATASET_BASE_PATH.name + "_hard_augmented")
    print(f"Saving augmented images at {aug_subdir_path}")
    sleep(3)
    for subdir_name in subdirs_names:
        (aug_subdir_path / subdir_name).mkdir(parents=True, exist_ok=True)

    # augment and save to disk
    pbar = tqdm(total=len(all_files), desc="Saving augmented images")

    with ProcessPoolExecutor() as executor:
        if DEBUG:
            print("DEBUG MODE: only testing 30 random images")
            all_files = random.sample(all_files, 30)
        futures = {executor.submit(augment_save_one_file, file, aug_subdir_path): file for file in all_files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                raise Exception(f"Error processing file {futures[future]}") from e
            pbar.update()

    pbar.close()

    # check result
    if not DEBUG:
        for subdir_name in subdirs_names:
            found_nb = len(list((aug_subdir_path / subdir_name).glob(f"*.{EXTENSION}")))
            expected_nb = 8 * len(list((DATASET_BASE_PATH / subdir_name).glob(f"*.{EXTENSION}")))
            assert found_nb == expected_nb, (
                f"Expected {expected_nb} files in {ending(aug_subdir_path / subdir_name, 2)}, found {found_nb}"
            )
        print("All checks passed")
