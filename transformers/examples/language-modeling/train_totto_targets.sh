export CUDA_VISIBLE_DEVICES=$1

python run_clm.py \
    --model_name_or_path "gpt2" \
    --output_dir "exp/totto_targets/gpt2" \
    --train_file "test_data/totto_targets/train.txt" \
    --validation_file "test_data/totto_targets/validation.txt" \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 12 \
    --evaluation_strategy "epoch" \
    --load_best_model_at_end \
    --save_total_limit 3 \
    --logging_steps 1000 \
    --do_train --do_eval