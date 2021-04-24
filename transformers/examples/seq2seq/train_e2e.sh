export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "t5-small" \
    --task "e2e" \
    --output_dir "exp/e2e/e2e_t5_small_01_test" \
    --train_file "test_data/e2e/train.json" \
    --validation_file "test_data/e2e/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 15 \
    --max_source_length 55 \
    --max_target_length 93 \
    --val_max_target_length 93 \
    --save_total_limit 8 \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --predict_with_generate \
    --num_beams 5 \
    --do_eval