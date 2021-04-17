export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "exp/e2e/e2e_k10_t5_small_01/checkpoint-3000" \
    --task "e2e" \
    --output_dir "exp/e2e/e2e_k10_t5_small_01" --overwrite_output_dir \
    --train_file "test_data/e2e_k10/train.json" \
    --validation_file "test_data/e2e_k10/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 6 \
    --max_source_length 148 \
    --max_target_length 93 \
    --val_max_target_length 93 \
    --save_total_limit 8 \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 3000 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval