from torch import float32
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, RandomHorizontalFlip, RandomVerticalFlip

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ContinuousTimeImageDataset, RandomRotationSquareSymmetry

DEFINITION = 196
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        ConvertImageDtype(float32),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotationSquareSymmetry(),
    ]
)

ds_params = DatasetParams(
    file_extension="png",
    key_transform=str,
    sorting_func=lambda subdir: float(subdir.name),
    dataset_class=ContinuousTimeImageDataset,
)

BBBC021_196_inference = DataSet(
    name="BBBC021_196_fully_ordered_inference",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
    fully_ordered=True,
)
