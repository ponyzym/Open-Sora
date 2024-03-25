num_frames = 16
frame_interval = 3
image_size = (512, 512)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=None,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/data/mazhiyuan/Open-Sora/pretrained_models/vae/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
<<<<<<< HEAD
    from_pretrained="./pretrained_models/t5",
=======
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
>>>>>>> upstream/main
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
