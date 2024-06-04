import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor


class VideoTimeEncoding(ModelMixin, ConfigMixin):
    """
    Class used to handle the video time encoding. Performs the following:

    1. Transforms the time float in [0;1] into a device'd Tensor
    2. Encodes that time information into a vector of sinusoidal bases with `diffusers.models.embeddings.Timesteps`:
    Hugging Face's reimplementation of the DDPM's reimplementation of the Transformer's sinusoidal positional encoding :)
    3. Passes the sinusoidal encoding through a smol 2-layers MLP (Hugging Face's `TimestepEmbedding`)

    Arguments
    ---------
    - encoding_dim (int): The number of dimensions returned by the sinusoidal encoding
    - time_embed_dim (int): The dimension of the diffusion timesteps embedding used in the denoiser UNet2DModel
    - flip_sin_to_cos (bool): ? TODO
    - downscale_freq_shift (float): ? TODO
    """

    @register_to_config
    def __init__(self, encoding_dim: int, time_embed_dim: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()

        def time_to_tensor(time: float, batch_size: int) -> Tensor:
            return torch.tensor([time] * batch_size, device=self.device)

        # returns a tensor or shape (batch_size)
        self.time_to_tensor = time_to_tensor

        # returns a tensor or shape (batch_size, encoding_dim)
        self.sinusoidal_encoding = Timesteps(
            num_channels=encoding_dim, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=downscale_freq_shift
        )

        self.learnable_time_embedding = TimestepEmbedding(in_channels=encoding_dim, time_embed_dim=time_embed_dim)

    def forward(self, time: float, batch_size: int) -> Tensor:
        """
        Encode the time information into a Tensor

        Arguments
        ---------
            - time (float): The video time to encode; assumed to be unique for each batch
            - batch_size (int): The number of elements in the batch

        Returns:
            video_time_embedding (Tensor): The time encoding tensor, shape: (batch_size, time_embed_dim)
        """
        time_tensor = self.time_to_tensor(time, batch_size)
        sinusoidal_code = self.sinusoidal_encoding(time_tensor)
        video_time_embedding = self.learnable_time_embedding(sinusoidal_code)
        return video_time_embedding
