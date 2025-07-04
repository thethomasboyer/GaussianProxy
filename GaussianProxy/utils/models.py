import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor


class VideoTimeEncoding(ModelMixin, ConfigMixin):
    """
    Class used to handle the video time encoding. Performs the following:

    1. Transforms the time float in [0;1] into a device'd Tensor if needed
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
    def __init__(
        self,
        encoding_dim: int,
        time_embed_dim: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
    ):
        super().__init__()

        # returns a tensor or shape (batch_size, encoding_dim)
        self.sinusoidal_encoding = Timesteps(
            num_channels=encoding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
        )

        self.learnable_time_embedding = TimestepEmbedding(in_channels=encoding_dim, time_embed_dim=time_embed_dim)

    def _time_to_tensor(self, time: int | float | Tensor, batch_size: int | None) -> Tensor:
        """Return a 1D tensor from `time`; no-op if already a Tensor."""
        if isinstance(time, Tensor):
            assert batch_size is None, f"`batch_size` must be `None` if `time` is a `Tensor`, got {batch_size}"
            assert time.ndim == 1, f"Expected `time` to be 1D, got {time.shape}"
            return time
        elif isinstance(time, (int, float)):
            assert batch_size is not None, "Must provide `batch_size` if `time` is a `float` or `int`"
            return torch.tensor([time] * batch_size, device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"`time` must be a `float` or a `Tensor`, got {type(time)}")

    def forward(self, time: int | float | Tensor, batch_size: int | None = None) -> Tensor:
        """
        Encode the time information.

        Arguments
        ---------
            - `time`: `int | float` | `Tensor`

        The video time to encode. Assumed to be unique for each batch if `float` or `int`,
        else the batch size is inferred from the passed `Tensor` shape, expected to be 1D
        (and `batch_size` must be `None` in this case).

            - `batch_size`: `int | None`

        The number of elements in the batch. Must be provided iff `time` is `float` or `int`.

        Returns
        -------
            video_time_embedding: `Tensor`

        The time encoding tensor, shape: (`batch_size`, `time_embed_dim`) where
        `batch_size` is either the passed `batch_size` or `time.shape[0]`
        """
        time_tensor = self._time_to_tensor(time, batch_size)
        sinusoidal_code = self.sinusoidal_encoding(time_tensor).to(time_tensor.dtype)
        video_time_embedding = self.learnable_time_embedding(sinusoidal_code)
        return video_time_embedding
