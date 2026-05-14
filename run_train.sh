#!/bin/bash

python /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/train.py \
  --config /home/bhavya_sai/Desktop/WGS_RFSPM-fixed-tx/configs/rfid-spectrum.yml \
  --gpu 0 \
  --mode train \
  --dataset_type rfid \
  --debug \
  --checkpoint_interval 100000