from GaussianProxy.conf.training_conf import TimeEncoderConfig, UNet2DConditionModelConfig

cross_attn_dim = 64

net = UNet2DConditionModelConfig(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(64, 128, 256),
    layers_per_block=2,
    act_fn="silu",
    cross_attention_dim=cross_attn_dim,
)

time_encoder = TimeEncoderConfig(
    encoding_dim=128,
    time_embed_dim=cross_attn_dim,
    flip_sin_to_cos=True,
    downscale_freq_shift=1,
)
