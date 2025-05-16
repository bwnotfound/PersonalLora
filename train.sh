export WANDB_PROJECT=PersonalLora
wandb login 输入wandb的api_key

export HF_ENDPOINT="https://hf-mirror.com"
export DS_SKIP_CUDA_CHECK=1

ds_config=config/ds_config.json

cat $ds_config
# DeepSpeed part
deepspeed --master_port 9001 train.py \
    --deepspeed $ds_config \
    --wandb_init false \
    --model_path /root/autodl-tmp/huggingface/models/Qwen2.5-0.5B-Instruct/ \
    --output_dir /root/autodl-tmp/output \
    --train_data_path data/train/movielens_llm/train.parquet \
    --eval_data_path data/train/movielens_llm/eval.parquet
