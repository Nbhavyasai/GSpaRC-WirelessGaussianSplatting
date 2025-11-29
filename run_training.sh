#!/bin/bash

#fixed_rx
python train.py --mode train --config configs/rfid-spectrum.yml --dataset_type rfid  --debug --eval_after_train model
