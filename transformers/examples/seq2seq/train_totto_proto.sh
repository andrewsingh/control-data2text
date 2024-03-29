export CUDA_VISIBLE_DEVICES=$1

python run_data_to_text.py \
    --model_name_or_path "t5-base" \
    --task "totto_proto" \
    --output_dir "exp/totto/totto_k5_t5_base" \
    --train_file "/projects/ogma2/users/andrewsi/control-data2text/transformers/examples/seq2seq/test_data/totto_proto/train_k5.json" \
    --validation_file "test_data/totto_proto/val_clean_source_embed.json" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 8 \
    --max_source_length 320 \
    --max_target_length 64 \
    --val_max_target_length 64 \
    --save_total_limit 6 \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 12000 \
    --load_best_model_at_end \
    --metric_for_best_model "F-score" \
    --predict_with_generate \
    --num_beams 5 \
    --do_train --do_eval