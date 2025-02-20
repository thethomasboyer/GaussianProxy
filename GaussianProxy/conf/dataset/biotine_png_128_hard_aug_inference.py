import torch
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset

DEFINITION = 128
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        ConvertImageDtype(torch.float32),
        Resize(size=DEFINITION),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

ds_params = DatasetParams(
    file_extension="png",
    key_transform=int,
    sorting_func=lambda subdir: int(subdir.name),
    dataset_class=ImageDataset,
)

dataset = DataSet(
    name="biotine_png_hard_aug",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=list(range(1, 20)),  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)
