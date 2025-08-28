# GIC-FAFNet

**GIC-FAFNet: Global-Local Information Coordination and Feature Alignment Fusion Network for Remote Sensing Object Detection**

This repository contains the implementation of **GIC-FAFNet**, built upon the [ultralytics](https://github.com/ultralytics/ultralytics) framework.  

## 📦 Requirements

This project depends on:

1. **Ultralytics framework dependencies**  
   Please follow the installation guide here: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

2. **Mamba model dependencies**  
   Please refer to: [https://github.com/MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)

> Experimental environment used in the paper:
> - `mamba_ssm==1.0.1`
> - `causal-conv1d==1.0.0`

## Training

Use train.py to train the GIC-FAFNet model:

python train.py --cfg configs/gic_fafnet.yaml --weights yolov8s.pt

## Validation / Batch Inference

Use val.py to validate the model or perform batch inference:

python val.py --weights runs/train/exp/weights/best.pt --data dataset.yaml

