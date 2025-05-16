import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from datasets.download import DownloadMode
from tqdm import tqdm
from utils import MovielensTextFormat, DataCollatorForMovielensTextLM
from module.lora import LoraLinear

if __name__ == "__main__":
    ckpt_path = "output/checkpoint-100"
    tokenizer_path = "output/tokenizer"
    eval_path = "data/train/movielens_llm/eval.parquet"

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    # print(tokenizer.eos_token)
    model = AutoPeftModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    device = "cuda"
    model.to(device)

    def ds_process_func(data_path):
        ds = load_dataset(
            "parquet",
            data_files=data_path,
            download_mode=(DownloadMode.REUSE_CACHE_IF_EXISTS),
        )["train"]
        ds.shuffle(seed=1437)
        ds = ds.map(
            MovielensTextFormat(
                tokenizer, max_length=model.base_model.config.max_position_embeddings
            ),
            batched=True,
            num_proc=1,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return ds

    eval_ds = ds_process_func(eval_path)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_ds, batch_size=4, collate_fn=DataCollatorForMovielensTextLM(tokenizer)
    )
    acc, total = 0, 0
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        user_indices = batch["user_indices"].squeeze().tolist()
        item_indices = batch["item_indices"].squeeze().tolist()
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    module.set_indices(user_indices, item_indices)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            logits = logits[:, -1, :]
            pred = torch.argmax(logits, dim=-1)
            acc += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"acc: {acc}, total: {total}, acc: {acc / total:.4f}")
