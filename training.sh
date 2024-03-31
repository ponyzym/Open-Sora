# 1 GPU, 16x256x256
# torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/opensora/train/16x256x256.py --data-path YOUR_CSV_PATH
# 4 GPUs, 64x512x512
torchrun --nnodes=1 --nproc_per_node=4 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT

# 2 nodes with --host arg, 64x512x512
# colossalai run --nproc_per_node 8 --host host1,host2 --master_addr host1 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
# N nodes with --hostfile arg, 64x512x512
# colossalai run --nproc_per_node 8 --hostfile ./hostfile --master_addr host1 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT


# only include certain hosts to execute commands
# colossalai run --nproc_per_node 8 --hostfile ./hostfile --master_addr host1  --include host1 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT
# exclude certain hosts to execute commands
# colossalai run --nproc_per_node 8 --hostfile ./hostfile --master_addr host1  --exclude host2 scripts/train.py configs/opensora/train/64x512x512.py --data-path YOUR_CSV_PATH --ckpt-path YOUR_PRETRAINED_CKPT