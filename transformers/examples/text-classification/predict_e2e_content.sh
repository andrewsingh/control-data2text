export CUDA_VISIBLE_DEVICES=$1

python run_glue.py \
    --model_name_or_path "exp/e2e_content/albert-base-v2/checkpoint-54520" \
    --output_dir $2 --overwrite_output_dir \
    --train_file "test_data/e2e_content/train.csv" \
    --validation_file "test_data/e2e_content/dev.csv" \
    --test_file $3  \
    --per_device_eval_batch_size 256 \
    --max_seq_length 128 \
    --do_predict
