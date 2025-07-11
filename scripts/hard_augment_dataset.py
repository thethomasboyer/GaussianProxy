# ruff: noqa: E402 # pylint: disable=wrong-import-position
print("Loading imports...", flush=True)
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from rich.traceback import install
from tqdm import tqdm

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
from GaussianProxy.utils.misc import generate_all_augs

###################################################### Arguments ######################################################
# fmt: off
DATASET_BASE_PATH = Path("/projects/static2dynamic/datasets/DiabeticRetinopathy/prepared_dataset/train")
EXTENSION         = "jpeg"
DEBUG             = True
CHECK_ONLY        = False
TRANSFORMS        = ["RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotationSquareSymmetry"]
AUG_SUBDIR_PATH   = DATASET_BASE_PATH.with_name(DATASET_BASE_PATH.name + "_hard_augmented")
N_THREADS         = 32
# fmt: on


####################################################### Helpers #######################################################
def ending(path: Path, n: int):
    return Path(*path.parts[-n:])


def augment_save_one_file(base_file: Path, aug_subdir_path: Path, in_parent_dir: bool):
    img = Image.open(base_file)
    augs = generate_all_augs(img, TRANSFORMS)
    for i, aug in enumerate(augs):
        # save where
        if in_parent_dir:
            this_aug_save_path = aug_subdir_path / base_file.parent.name / f"{base_file.stem}_aug_{i}.{EXTENSION}"
        else:
            this_aug_save_path = aug_subdir_path / f"{base_file.stem}_aug_{i}.{EXTENSION}"
        # save (or not)
        if not DEBUG:
            aug.save(this_aug_save_path)


######################################################### Main ########################################################
if __name__ == "__main__":
    install()

    print(f"Augmenting base dataset located at: {DATASET_BASE_PATH}", flush=True)
    print(f"Augmented dataset will be saved at: {AUG_SUBDIR_PATH}", flush=True)
    print(f"Using the following transforms:     {TRANSFORMS}", flush=True)
    print("DEBUG:                             ", DEBUG, flush=True)
    print("CHECK_ONLY:                        ", CHECK_ONLY, flush=True)
    print("N_THREADS:                         ", N_THREADS, flush=True)
    inpt = input("Proceed? (y/[n])")
    if inpt.lower() != "y":
        print("Exiting")
        sys.exit(0)

    # get names of subdirs / timestamps
    subdirs_names = [x.name for x in DATASET_BASE_PATH.iterdir() if x.is_dir()]
    print(f"Found {len(subdirs_names)} subdirectories: {subdirs_names}", flush=True)

    # get all base files to augment
    print(f"Searching for files with extension '{EXTENSION}' in '{DATASET_BASE_PATH}' ...", flush=True)
    all_files = list(DATASET_BASE_PATH.rglob(f"*.{EXTENSION}", recurse_symlinks=True))
    print(f"Found {len(all_files)} base files in total", flush=True)
    in_parent_dir = len(subdirs_names) != 0

    # create augmented subdirs in a adjacent dir to the base dataset one
    if not CHECK_ONLY:
        inpt = input(f"Saving augmented images at {AUG_SUBDIR_PATH}: proceed? (y/[n])")
        if inpt.lower() != "y":
            print("Exiting")
            sys.exit(0)
        if len(subdirs_names) != 0:
            for subdir_name in subdirs_names:
                (AUG_SUBDIR_PATH / subdir_name).mkdir(parents=True, exist_ok=True)
        else:
            AUG_SUBDIR_PATH.mkdir(parents=True, exist_ok=True)

        # augment and save to disk
        if DEBUG:
            print("DEBUG MODE: only testing 30 random images")
            all_files = random.sample(all_files, 30)
        pbar = tqdm(total=len(all_files), desc="Saving augmented images")

        with ThreadPoolExecutor(N_THREADS) as executor:
            futures = {
                executor.submit(augment_save_one_file, file, AUG_SUBDIR_PATH, in_parent_dir): file for file in all_files
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise RuntimeError(f"Error processing file {futures[future]}") from e
                pbar.update()

        pbar.close()

    # check result
    if not DEBUG:
        if len(subdirs_names) != 0:
            for subdir_name in subdirs_names:
                found_nb = len(list((AUG_SUBDIR_PATH / subdir_name).glob(f"*.{EXTENSION}")))
                expected_nb = (2 ** len(TRANSFORMS)) * len(
                    list((DATASET_BASE_PATH / subdir_name).glob(f"*.{EXTENSION}"))
                )
                assert found_nb == expected_nb, (
                    f"Expected {expected_nb} files in {ending(AUG_SUBDIR_PATH / subdir_name, 2)}, found {found_nb}"
                )
        else:
            found_nb = len(list(AUG_SUBDIR_PATH.glob(f"*.{EXTENSION}")))
            expected_nb = (2 ** len(TRANSFORMS)) * len(list(DATASET_BASE_PATH.glob(f"*.{EXTENSION}")))
            assert found_nb == expected_nb, (
                f"Expected {expected_nb} files in {ending(AUG_SUBDIR_PATH, 2)}, found {found_nb}"
            )
        print("All checks passed")
