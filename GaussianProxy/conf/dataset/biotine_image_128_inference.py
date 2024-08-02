import torchvision
from conf.training_conf import DataSet

DEFINITION = 128
NUMBER_OF_CHANNELS = 3

transforms = torchvision.transforms.transforms.Compose(
    transforms=[
        torchvision.transforms.Normalize(mean=[0.5] * NUMBER_OF_CHANNELS, std=[0.5] * NUMBER_OF_CHANNELS),
        torchvision.transforms.transforms.Resize(size=DEFINITION),
    ]
)

biotine_image_inference = DataSet(
    name="biotine_image",
    data_shape=(NUMBER_OF_CHANNELS, DEFINITION, DEFINITION),
    transforms=transforms,
    selected_dists=list(range(1, 20)),  # not used
    expected_initial_data_range=(0, 1),
)
