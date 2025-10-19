#!/bin/bash

code_dir=DSpAST

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

blr=1e-3
mask_t_prob=0.25
mask_f_prob=0.25

# Download from ...
ckpt=/path_to_models/tp78yk/models/finetune-stage2.pth
# or use your own checkpoint:
ckpt=./outputs/finetune-stage1/checkpoint-99.pth

# Sound source
dataset=audioset
audio_path_root=/path_to_data/SpatialSoundQA/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet
audioset_label=/path_to_data/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv
audioset_train_json=/path_to_data/SpatialSoundQA/AudioSet/metadata/unbalanced.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/unbalanced.json
audioset_train_weight=/path_to_data/SpatialSoundQA/AudioSet/metadata/weights/unbalanced_weight.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/weights/unbalanced_weight.csv
audioset_eval_json=/path_to_data/SpatialSoundQA/AudioSet/metadata/eval.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/eval.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_type=binaural # or mono
reverb_path_root=/path_to_data/SpatialSoundQA/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_train_json=/path_to_data/SpatialSoundQA/mp3d_reverb/train_reverberation.json
reverb_val_json=/path_to_data/SpatialSoundQA/mp3d_reverb/eval_reverberation.json

# logging path
output_dir=./outputs/finetune-stage2
log_dir=$output_dir/log

mkdir -p $output_dir
python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=21373 --use_env $code_dir/main_finetune_dspast.py \
    --log_dir $log_dir --output_dir $output_dir --finetune $ckpt \
    --model build_AST --dataset $dataset \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label \
    --nb_classes 355 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --blr $blr --dist_eval --batch_size 32 --num_workers 8 \
    --roll_mag_aug --mixup 0.5 --audio_normalize \
    --mask_t_prob $mask_t_prob --mask_f_prob $mask_f_prob \
    --first_eval_ep 20 --epochs 50 --warmup_epochs 10 \
    --sed_weight 100 --dp_weight 2 --doae_weight 1 \
    --mask_2d
