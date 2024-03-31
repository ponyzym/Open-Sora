num_frames = 360
frame_interval = 1
image_size = (512, 512)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2-seq"
sp_size = 2

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    from_pretrained=None,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    enable_sequence_parallelism=True,  # enable sq here
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/data/mazhiyuan/Open-Sora/pretrained_models/vae/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "results"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 250
load = None

batch_size = 1
lr = 2e-5
grad_clip = 1.0
