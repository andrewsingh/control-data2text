export CUDA_VISIBLE_DEVICES=$1

python run_clm.py \
    --model_name_or_path "results/e2e_targets/gpt2-02/checkpoint-10140" \
    --output_dir "results/e2e_targets/gpt2-02" --overwrite_output_dir \
    --train_file "test_data/e2e_targets/train.txt" \
    --validation_file "test_data/e2e_targets/validation.txt" \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 25 \
    --evaluation_strategy "epoch" \
    --save_total_limit 10 \
    --logging_steps 500 \
    --do_train --do_eval