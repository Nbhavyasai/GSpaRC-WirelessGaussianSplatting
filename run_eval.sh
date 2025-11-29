# #!/bin/bash

# Test 

python train.py \
  --mode eval \
  --config configs/rfid-spectrum.yml \
  --dataset_type rfid \
  --gpu 0 \
  --load_checkpoint "path" \
  --eval_after_train off
