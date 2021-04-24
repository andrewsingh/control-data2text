export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "facebook/bart-base" \
    --task "totto" \
    --output_dir "exp/totto/totto_bart_base" \
    --train_file "test_data/totto/train.json" \
    --validation_file "test_data/totto/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 20 \
    --max_source_length 256 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --save_total_limit 8 \
    --logging_steps 500 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval