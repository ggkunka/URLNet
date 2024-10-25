#!/bin/bash

# Train the model
python train.py \
    --data_data_dir data/train_converted.txt \
    --data_dev_pct 0.1 \
    --data_delimit_mode 0 \
    --data_min_word_freq 1 \
    --model_emb_mode 1 \
    --model_emb_dim 32 \
    --model_filter_sizes 3,4,5,6 \
    --train_nb_epochs 5 \
    --train_batch_size 32 \
    --log_output_dir runs/1000_emb1_dlm0_run/

# Test the model
python test.py \
    --data_data_dir data/train_converted.txt \
    --data_delimit_mode 0 \
    --data_word_dict_dir runs/1000_emb1_dlm0_run/words_dict.p \
    --data_subword_dict_dir runs/1000_emb1_dlm0_run/subwords_dict.p \
    --data_char_dict_dir runs/1000_emb1_dlm0_run/chars_dict.p \
    --log_checkpoint_dir runs/1000_emb1_dlm0_run/checkpoints/ \
    --log_output_dir runs/1000_emb1_dlm0_run/ \
    --model_emb_mode 1 \
    --model_emb_dim 32 \
    --test_batch_size 32


python auc.py --input_path runs/${train_size}_emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ --input_file test_results.txt --threshold 0.5
done
