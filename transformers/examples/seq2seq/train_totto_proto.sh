export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "exp/totto/totto_k10_max10_t5_small/checkpoint-300000" \
    --task "totto_proto" \
    --output_dir "exp/totto/totto_k10_max10_t5_small" --overwrite_output_dir \
    --train_file "test_data/totto_proto/train_k10_max10.json" \
    --validation_file "test_data/totto_proto/val_headers_only.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 12 \
    --max_source_length 320 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --save_total_limit 20 \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 15000 \
    --load_best_model_at_end \
    --metric_for_best_model "F-score" \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval