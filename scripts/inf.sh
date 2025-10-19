#!/bin/bash

code_dir=DSpAST

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

dataset=audioset
ckpt=./path_to_model/checkpoint.pth

# Sound source
dataset=audioset
audio_path_root=/path_to_data/SpatialSoundQA/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet
audioset_label=/path_to_/data/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv
audioset_train_json=/path_to_data/SpatialSoundQA/AudioSet/metadata/balanced.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/balanced.json
audioset_train_weight=/path_to_data/SpatialSoundQA/AudioSet/metadata/weights/balanced_weight.csv # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/weights/balanced_weight.csv
audioset_eval_json=/path_to_data/SpatialSoundQA/AudioSet/metadata/eval.json # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/eval.json

# For reverberation data, please visit https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_type=binaural # binaural or mono
reverb_path_root=/path_to_data/SpatialSoundQA/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
reverb_train_json=/path_to_data/SpatialSoundQA/mp3d_reverb/train_reverberation.json
reverb_val_json=/path_to_data/SpatialSoundQA/mp3d_reverb/eval_reverberation.json

# logging path
output_dir=./outputs/eval
log_dir=./outputs/eval/log


python -m torch.distributed.launch \
    --nproc_per_node=1 --use_env $code_dir/main_finetune.py \
    --log_dir ${log_dir} --output_dir ${output_dir} \
    --model build_AST --dataset $dataset --finetune $ckpt \
    --audio_path_root $audio_path_root \
    --audioset_train $audioset_train_json --audioset_eval $audioset_eval_json \
    --label_csv $audioset_label --nb_classes 355 \
    --reverb_path_root $reverb_path_root --reverb_type $reverb_type \
    --reverb_train $reverb_train_json --reverb_val $reverb_val_json \
    --batch_size 64 --num_workers 4 \
    --audio_normalize \
    --eval --dist_eval
