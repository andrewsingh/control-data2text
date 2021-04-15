export CUDA_VISIBLE_DEVICES=$1

python run_clm.py \
    --model_name_or_path "gpt2" \
    --output_dir "results/e2e_targets/gpt2-01" \
    --train_file "test_data/e2e_targets/train.txt" \
    --validation_file "test_data/e2e_targets/val.txt" \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 40 \
    --evaluation_strategy "epoch" \
    --load_best_model_at_end \
    --logging_steps 100 \
    --do_train --do_eval