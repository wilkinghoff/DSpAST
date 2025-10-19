#!/bin/bash

code_dir=Spatial-AST-main

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

# Download from https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link
ckpt=/home/es.aau.dk/tp78yk/models/spatial-ast/pretrained.pth

# Sound source
dataset=audioset
audio_path_root=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet
audioset_label=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv
audioset_train_json=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/AudioSet/metadata/unbalanced.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/unbalanced.json
audioset_train_weight=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/AudioSet/metadata/weights/unbalanced_weight.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/weights/unbalanced_weight.csv
audioset_eval_json=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/AudioSet/metadata/eval.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/eval.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_type=binaural # or mono
reverb_path_root=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_train_json=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/mp3d_reverb/train_reverberation.json
reverb_val_json=/home/es.aau.dk/tp78yk/data/SpatialSoundQA/mp3d_reverb/eval_reverberation.json

# logging path
output_dir=./outputs/finetune-stage1
log_dir=$output_dir/log

mkdir -p $output_dir
python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=52741 --use_env $code_dir/main_finetune_dspast.py \
    --log_dir $log_dir --output_dir $output_dir --finetune $ckpt \
    --model build_AST --dataset $dataset \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label --weight_csv $audioset_train_weight \
    --nb_classes 355 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --blr $blr --dist_eval --batch_size 32 --num_workers 8 \
    --roll_mag_aug --mixup 0.5 --audio_normalize \
    --mask_t_prob $mask_t_prob --mask_f_prob $mask_f_prob \
    --first_eval_ep 0 --epochs 100 --warmup_epochs 10 --epoch_len 186175 \
    --sed_weight 1 --dp_weight 0 --doae_weight 0 \
    --weight_sampler --distributed_wrapper --mask_2d
