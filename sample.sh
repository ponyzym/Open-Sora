# Sample 16x256x256 (5s/sample, 100 time steps, 22 GB memory)
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path pretrained_models/opensora/OpenSora-v1-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt
# Auto Download
# torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x256x256.py --ckpt-path pretrained_models/opensora/OpenSora-v1-HQ-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 16x512x512 (20s/sample, 100 time steps, 24 GB memory)
# torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x512x512.py --ckpt-path pretrained_models/opensora/OpenSora-v1-HQ-16x512x512.pth --prompt-path ./assets/texts/t2v_samples.txt
# Auto Download
# torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/16x512x512.py --ckpt-path pretrained_models/opensora/OpenSora-v1-HQ-16x512x512.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 64x512x512 (40s/sample, 100 time steps)
# torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/opensora/inference/64x512x512.py --ckpt-path pretrained_models/opensora/OpenSora-v1-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt

# Sample 64x512x512 with sequence parallelism (30s/sample, 100 time steps)
# sequence parallelism is enabled automatically when nproc_per_node is larger than 1
# torchrun --standalone --nproc_per_node 2 scripts/inference.py configs/opensora/inference/64x512x512.py --ckpt-path pretrained_models/opensora/OpenSora-v1-16x256x256.pth --prompt-path ./assets/texts/t2v_samples.txt