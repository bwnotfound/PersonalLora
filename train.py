import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from functools import partial
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from utils import (
    DataCollatorForMovielensTextLM,
    MovielensTextFormat,
    MovieLensAccuracy,
)
from module.lora import LoraLinear
from datasets.download import DownloadMode


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        user_indices = inputs.pop("user_indices")
        assert (
            user_indices.size(0) == 1
        ), f"目前只支持每batch单用户训练, {user_indices.shape}"
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                if hasattr(module, "set_user_index"):
                    module.set_user_index(user_indices.squeeze().item())
        outputs = model(**inputs)

        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
        )

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def load_mt(path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def train_model(
    model_path,
    output_dir,
    train_data_path,
    eval_data_path=None,
    tokenizer_path=None,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    config = {
        "output_dir": output_dir,
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "logging_strategy": "steps",
        "logging_dir": "logs",
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 300,
        "eval_strategy": "no",
        "save_total_limit": 3,
        "learning_rate": 2e-5,
        "weight_decay": 1e-4,
        "gradient_accumulation_steps": 512,
        "gradient_checkpointing": False,
        "save_only_model": True,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "bf16": True,
    }
    if eval_data_path is None:
        config["eval_strategy"] = "no"
    import wandb
    from datetime import datetime

    assert os.path.exists("wandb.key"), "Please put your wandb key in wandb.key"
    with open("wandb.key") as f:
        wandb.login(key=f.read().strip())
    wandb.init(
        project="PersonalLora",
        config=config,
        name="-{}".format(
            os.path.basename(model_path), datetime.now().strftime("%Y-%m-%d")
        ),
    )

    model, tokenizer = load_mt(model_path, tokenizer_path=tokenizer_path)

    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    custom_module_mapping = {nn.Linear: partial(LoraLinear, num_users=10000)}
    peft_config._register_custom_module(custom_module_mapping)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    use_dataset_cache = True

    def ds_process_func(data_path):
        ds = load_dataset(
            "parquet",
            data_files=data_path,
            download_mode=(
                DownloadMode.REUSE_CACHE_IF_EXISTS if not use_dataset_cache else None
            ),
        )["train"]
        ds.shuffle()
        ds = ds.map(
            MovielensTextFormat(
                tokenizer, max_length=model.config.max_position_embeddings
            ),
            batched=True,
            num_proc=4,
            remove_columns=ds.column_names,
        )
        return ds

    train_ds = ds_process_func(train_data_path)
    if eval_data_path:
        eval_ds = ds_process_func(eval_data_path)
    else:
        eval_ds = None

    training_args = TrainingArguments(**config)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForMovielensTextLM(tokenizer),
        compute_metrics=MovieLensAccuracy(tokenizer).compute_metrics,
    )
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    trainer.train()


if __name__ == "__main__":
    train_model(
        # "Qwen/Qwen2.5-0.5B-Instruct",
        "/root/autodl-tmp/huggingface/models/Qwen2.5-0.5B-Instruct/",
        "/root/autodl-tmp/output",
        "data/train/movielens_llm/train.parquet",
        "data/train/movielens_llm/eval.parquet",
    )
