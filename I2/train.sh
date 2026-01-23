#!/bin/bash
python train_ldm.py \
    --mri_dir D:\zxy\paper\zxydata\zxyself_traindata\AD_NC\MRI_Normalized \
    --pet_dir D:\zxy\paper\zxydata\zxyself_traindata\AD_NC\PET_Normalized \
    --csv_path D:\zxy\paper\zxydata\zxyself_traindata\AD_NC\mri_2_pet_mapping.csv \
    --num_epochs 100 \
    --batch_size 2 \
    --learning_rate 0.00005 \
    --grad_accum_steps 8 \
    --timesteps 600
