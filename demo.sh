#!/bin/bash

emb_modes=(1 2 2 3 3 4 5)
delimit_modes=(0 0 1 0 1 1 1)
train_size=100000
test_size=100000
nb_epoch=5

for ((i=0; i < ${#emb_modes[@]}; ++i)); do
    python train.py \
        --data_data_dir data/train_converted.txt \
        --data_dev_pct 0.001 \
        --data_delimit_mode ${delimit_modes[$i]} \
        --data_min_word_freq 1 \
        --model_emb_mode ${emb_modes[$i]} \
        --model_emb_dim 32 \
        --model_filter_sizes 3,4,5,6 \
        --train_nb_epochs ${nb_epoch} \
        --train_batch_size 1048 \
        --log_print_every 5 \
        --log_eval_every 10 \
        --log_checkpoint_every 10 \
        --log_output_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/

    python test.py \
        --data_data_dir data/train_converted.txt \
        --data_delimit_mode ${delimit_modes[$i]} \
        --data_word_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/words_dict.p \
        --data_subword_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/subwords_dict.p \
        --data_char_dict_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/chars_dict.p \
        --log_checkpoint_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/checkpoints/ \
        --log_output_dir runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ \
        --model_emb_mode ${emb_modes[$i]} \
        --model_emb_dim 32 \
        --test_batch_size 1048

    python auc.py \
        --input_path runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ \
        --input_file test_results.txt \
        --threshold 0.5
done
