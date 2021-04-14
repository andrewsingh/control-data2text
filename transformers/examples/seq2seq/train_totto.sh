export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "t5-small" \
    --output_dir "exp/totto/baseline_t5_small" \
    --train_file "test_data/totto/train.json" \
    --validation_file "test_data/totto/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 20 \
    --max_source_length 256 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --save_total_limit 8 \
    --logging_steps 1000 \
    --load_best_model_at_end \
    --evaluation_strategy "epoch" \
    --predict_with_generate \
    --do_eval