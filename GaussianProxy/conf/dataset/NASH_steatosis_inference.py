from torch import float32
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset, RandomRotationSquareSymmetry

CROP_SIZE = 192
RESIZED_FINAL_DEFINITION = 128
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        # 1. convert to float32 (and normalize to [0, 1])
        # before resizing
        ConvertImageDtype(float32),
        # 2. random crop from 299x299 to 192x192 (41% of area), then resize to 128x128
        RandomCrop(CROP_SIZE),
        Resize(RESIZED_FINAL_DEFINITION),
        # 3. normalize to [-1, 1]
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
        # 4. random 8x square augmentations
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotationSquareSymmetry(),
    ]
)

ds_params = DatasetParams(
    file_extension="png",
    key_transform=int,
    sorting_func=lambda subdir: int(subdir.name),
    dataset_class=ImageDataset,
)

dataset = DataSet(
    name="NASH_steatosis",
    data_shape=(NUMBER_OF_CHANNELS, RESIZED_FINAL_DEFINITION, RESIZED_FINAL_DEFINITION),
    transforms=transforms,
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)
