import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from functools import partial
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from accelerate.utils import DistributedType
from utils import (
    DataCollatorForMovielensTextLM,
    MovielensTextFormat,
    MovieLensAccuracy,
    preprocess_logits_for_metrics,
)
from module.lora import LoraLinear
from datasets.download import DownloadMode


@dataclass
class ScriptArguments:
    local_rank: int = field(
        default=0,
        metadata={"help": "Used for multi-gpu"},
    )
    num_users: int = field(
        default=6050,
        metadata={"help": "Number of users."},
    )
    num_items: int = field(
        default=4000,
        metadata={"help": "Number of items."},
    )
    lora_r: int = field(
        default=4,
        metadata={"help": "Lora rank."},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "Lora alpha."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."},
    )
    num_procs_for_ds_process: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config file."},
    )
    wandb_init: bool = field(
        default=True,
        metadata={
            "help": "Whether to init wandb in program. If not, please init wandb in shell script."
        },
    )
    model_path: str = field(
        default="/root/autodl-tmp/huggingface/models/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path to the model."},
    )
    output_dir: str = field(
        default="/root/autodl-tmp/output",
        metadata={"help": "Path to the output directory."},
    )
    train_data_path: str = field(
        default="data/train/movielens_llm/train.parquet",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: Optional[str] = field(
        default="data/train/movielens_llm/eval.parquet",
        metadata={"help": "Path to the evaluation data."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenizer."},
    )


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        user_indices = inputs.pop("user_indices").squeeze().tolist()
        item_indices = inputs.pop("item_indices").squeeze().tolist()
        # assert (
        #     user_indices.size(0) == 1
        # ), f"目前只支持每batch单用户训练, {user_indices.shape}"
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                module.set_indices(user_indices, item_indices)
        outputs = model(**inputs)

        logits = outputs.logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            reduction="mean",
        )

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        else:
            loss = loss / self.args.gradient_accumulation_steps
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()


def load_mt(path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def train_model():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model_path = args.model_path
    output_dir = args.output_dir
    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path
    tokenizer_path = args.tokenizer_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    config = {
        "output_dir": output_dir,
        "num_train_epochs": 100,
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 128,
        "logging_strategy": "steps",
        "logging_dir": "logs",
        "logging_steps": 1,
        "save_steps": 100,
        "eval_steps": 100,
        "eval_strategy": "steps",
        "save_total_limit": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_accumulation_steps": 128, # gradient_accumulation_steps = 1024 / n_gpus
        "eval_accumulation_steps": 1,
        # "batch_eval_metrics": True,
        "gradient_checkpointing": False,
        "save_only_model": True,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "bf16": True,
        "deepspeed": args.deepspeed,
    }
    if eval_data_path is None:
        config["eval_strategy"] = "no"

    if args.wandb_init:
        import wandb
        from datetime import datetime

        assert os.path.exists("wandb.key"), "Please put your wandb key in wandb.key"
        with open("wandb.key") as f:
            wandb.login(key=f.read().strip())
        wandb.init(
            project="PersonalLora",
            config=config,
            name="{}-{}".format(
                os.path.basename(model_path.rstrip("/")),
                datetime.now().strftime("%Y-%m-%d"),
            ),
        )

    model, tokenizer = load_mt(model_path, tokenizer_path=tokenizer_path)

    def ds_process_func(data_path):
        ds = load_dataset(
            "parquet",
            data_files=data_path,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        )["train"]
        # ds.shuffle(seed=1437)
        ds = ds.map(
            MovielensTextFormat(
                tokenizer, max_length=model.config.max_position_embeddings
            ),
            batched=True,
            num_proc=args.num_procs_for_ds_process,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return ds

    train_ds = ds_process_func(train_data_path)
    if eval_data_path:
        eval_ds = ds_process_func(eval_data_path)
    else:
        eval_ds = None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    custom_module_mapping = {
        nn.Linear: partial(
            LoraLinear, num_users=args.num_users, num_items=args.num_items
        )
    }
    peft_config._register_custom_module(custom_module_mapping)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(**config)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForMovielensTextLM(tokenizer),
        compute_metrics=MovieLensAccuracy(tokenizer).compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    trainer.train()


if __name__ == "__main__":
    train_model()
