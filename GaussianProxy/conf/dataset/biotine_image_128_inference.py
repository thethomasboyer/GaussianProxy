from torchvision.transforms import Compose, Normalize, Resize

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import NumpyDataset

DEFINITION = 128
NUMBER_OF_CHANNELS = 3

transforms = Compose(
    transforms=[
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
        Resize(size=DEFINITION),
    ]
)

ds_params = DatasetParams(
    file_extension="npy",
    key_transform=int,
    sorting_func=lambda subdir: int(subdir.name),
    dataset_class=NumpyDataset,
)

biotine_image_inference = DataSet(
    name="biotine_image",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=list(range(1, 20)),  # not used
    expected_initial_data_range=(0, 1),
    dataset_params=ds_params,
)
