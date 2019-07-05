#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=2 main.py --config=UNet3D/config/config.yaml --is_distributed