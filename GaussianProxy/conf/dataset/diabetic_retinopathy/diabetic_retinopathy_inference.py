from torch import float32
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.v2 import CenterCrop

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset

DEFINITION = 256
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        # single int => image resized to (size * aspect_ratio, size) or (size, size * aspect_ratio) with aspect_ratio >= 1 preserved
        Resize(256),
        CenterCrop(256),  # pyright: ignore[reportAttributeAccessIssue]
        ConvertImageDtype(float32),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

ds_params = DatasetParams(
    file_extension="jpeg",
    key_transform=int,
    sorting_func=lambda subdir: int(subdir.name),
    dataset_class=ImageDataset,
)

diabetic_retinopathy_inference = DataSet(
    name="diabetic_retinopathy",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)
