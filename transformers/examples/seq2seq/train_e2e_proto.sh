export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "t5-small" \
    --task "e2e_proto" \
    --output_dir "exp/e2e/e2e_k3_no_mask_t5_small" \
    --train_file "test_data/e2e_proto/train_k3_no_mask.json" \
    --validation_file "test_data/e2e_proto/validation.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 12 \
    --max_source_length 148 \
    --max_target_length 93 \
    --val_max_target_length 93 \
    --save_total_limit 4 \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 5000 \
    --load_best_model_at_end \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval