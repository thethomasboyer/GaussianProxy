from torch import float32
from torchvision.transforms import Compose, ConvertImageDtype, Normalize

from GaussianProxy.conf.training_conf import DataSet, DatasetParams
from GaussianProxy.utils.data import ImageDataset

DEFINITION = 66
NUMBER_OF_CHANNELS = 1

transforms = Compose(
    transforms=[
        ConvertImageDtype(float32),
        Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
    ]
)

phase_order = (
    "G1",
    "S",
    "G2",
    "Prophase",
    "Metaphase",
    "Anaphase",
    "Telophase",
)
phase_order_dict = {phase: index for index, phase in enumerate(phase_order)}
ds_params = DatasetParams(
    file_extension="png",
    key_transform=str,
    sorting_func=lambda subdir: phase_order_dict[subdir.name],
    dataset_class=ImageDataset,
)

dataset = DataSet(
    name="Jurkat_brightfield",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=None,  # not used
    expected_initial_data_range=(0, 255),
    dataset_params=ds_params,
)
