export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "t5-small" \
    --task "e2e_dtg_si" \
    --loss_lambda 1 \
    --output_dir "exp/e2e/e2e_dtg_si_t5_small_lambda_1" \
    --train_file "test_data/e2e_dtg_si/train.json" \
    --validation_file "test_data/e2e_dtg_si/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 15 \
    --max_source_length 102 \
    --max_target_length 152 \
    --val_max_target_length 93 \
    --save_total_limit 8 \
    --logging_steps 500 \
    --evaluation_strategy "steps" \
    --eval_steps 3000 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval