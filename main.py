import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
eos_token = "<|end_of_text|>"


def load_mt(path, to=True, bf16=False, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = path
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16 if bf16 else torch.float32
    )
    if not os.path.exists(path):
        tokenizer.add_special_tokens({"eos_token": eos_token})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        elif tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token = None
            tokenizer.pad_token_id = None
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        assert tokenizer.eos_token_id != tokenizer.pad_token_id
        print(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        for index in tokenizer.additional_special_tokens:
            model.get_input_embeddings().weight.data[
                index
            ] = model.get_input_embeddings().weight.data[tokenizer.eos_token_id]
    if to:
        model.to(device)
    return model, tokenizer


def train_model(data_path="data/raw/COIG_default.parquet", data_dir="."):
    data_path = os.path.join(data_dir, data_path)
    config = {
        "output_dir": os.path.join(data_dir, "output"),
        "num_train_epochs": 4,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "logging_dir": "logs",
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 300,
        "evaluation_strategy": "steps",
        "save_total_limit": 3,
        "learning_rate": 2e-5,
        "weight_decay": 1e-4,
        "gradient_accumulation_steps": 128,
        "gradient_checkpointing": False,
        "save_only_model": True,
        "report_to": "wandb",
        "bf16": False,
        "use_cpu": False,
    }
    resume_from_last = False
    import wandb
    from datetime import datetime

    assert os.path.exists("wandb.key"), "Please put your wandb key in wandb.key"
    with open("wandb.key") as f:
        wandb.login(key=f.read().strip())
    wandb.init(
        project="PersonalLora",
        config=config,
        name="GPT-Neo-125M-{}".format(datetime.now().strftime("%Y-%m-%d")),
    )

    from datasets import load_dataset, Dataset
    from utils import (
        DataCollatorForSessionLM,
        CLMBleu,
        compute_loss_func,
        RowFormat,
    )

    model, tokenizer = load_mt("EleutherAI/gpt-neo-125m", to=False, bf16=config["bf16"])

    RELOAD = True

    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "data/eval"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "data/train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)

    data_name = ".".join(os.path.basename(data_path).split(".")[:-1])
    train_path = os.path.join(data_dir, f"data/train/{data_name}_train.parquet")
    eval_path = os.path.join(data_dir, f"data/eval/{data_name}_eval.parquet")

    if RELOAD and os.path.exists(train_path) and os.path.exists(eval_path):
        train_ds = Dataset.load_from_disk(train_path)
        eval_ds = Dataset.load_from_disk(eval_path)
    else:
        ds = load_dataset("parquet", data_files=data_path)
        ds = ds.shuffle()
        # ds["train"] = ds["train"].select(range(10000))

        ds = ds.map(
            RowFormat(tokenizer, max_length=model.config.max_position_embeddings),
            batched=True,
            num_proc=4,
            remove_columns=ds["train"].column_names,
        )
        eval_ratio = 0.1
        '''
        先shuffle，然后切分eval_ratio的数据作为验证集，剩下的作为训练集。保存在data/train和data/eval文件夹中。    
        '''
        ds = ds["train"].train_test_split(test_size=int(len(ds["train"]) * eval_ratio))
        train_ds, eval_ds = ds["train"], ds["test"]
        train_ds.save_to_disk(train_path)
        eval_ds.save_to_disk(eval_path)

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(**config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSessionLM(
            tokenizer,
            max_length=model.config.max_position_embeddings,
        ),
        compute_metrics=CLMBleu(tokenizer).compute_metrics,
        compute_loss_func=compute_loss_func,
    )
    tokenizer.save_pretrained(os.path.join(data_dir, "output/tokenizer"))
    print(f"max_length: {model.config.max_position_embeddings}")
    if not resume_from_last:
        trainer.train()
    else:
        try:
            trainer.train(resume_from_last)
        except Exception as e:
            print(f"resume_from_last error. Try from pretrained")
            trainer.train()



def chat_app(model=None, tokenizer=None, port=None):
    if any(x is None for x in [model, tokenizer]):
        assert all(x is None for x in [model, tokenizer])
        model, tokenizer = load_mt("EleutherAI/gpt-neo-125m")
    else:
        assert all(x is not None for x in [model, tokenizer])
        if isinstance(model, str):
            model, tokenizer = load_mt(model, tokenizer_path=tokenizer)
    import gradio as gr
    from transformers import GenerationConfig

    def generate_text(prompt, max_length=500):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        generate_config = GenerationConfig(
            do_sample=False,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs = model.generate(
            **inputs,
            generation_config=generate_config,
            
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    '''
    界面包含输入框和输出框以及一个按钮，点击按钮后会调用generate_text函数，将输入框中的文本传入函数，函数返回的文本会显示在输出框中。
    '''
    with gr.Blocks() as block:
        input_text = gr.Textbox(lines=5, label="Input Text")
        max_length_int = gr.Slider(minimum=1, maximum=1000, value=500, label="Max Length")
        output_text = gr.Textbox(lines=5, label="Output Text")
        button = gr.Button("Generate")
        # button.click(generate_text, inputs=input_text, outputs=output_text)
        button.click(generate_text, inputs=[input_text, max_length_int], outputs=output_text)
    block.launch(server_port=port)


def eval_simple_test():
    from tqdm import tqdm
    from datasets import load_dataset
    from utils import RowFormat, DataCollatorForSessionLM

    model_path = "output/checkpoint-280"
    tokenizer_path = "output/tokenizer"
    batch_size = 4
    model, tokenizer = load_mt(model_path, tokenizer_path=tokenizer_path)

    ds = load_dataset("parquet", data_files="data/raw/simple_test/eval.parquet")

    ds = ds.map(
        RowFormat(tokenizer, max_length=model.config.max_position_embeddings),
        batched=True,
        num_proc=4,
        remove_columns=ds["train"].column_names,
    )
    eval_ds = ds["train"]

    data_collator = DataCollatorForSessionLM(
        tokenizer,
        max_length=model.config.max_position_embeddings,
    )

    acc_cnt = 0
    total = 0

    for i in tqdm(range(0, len(eval_ds), batch_size)):
        batch = eval_ds[i : i + batch_size]
        inputs = data_collator(batch)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        preds = logits.argmax(dim=-1)
        acc_cnt += (preds == labels).sum().item()
        total += labels.ne(-100).sum().item()
    assert total > 0
    print(f"Accuracy: {acc_cnt / total * 100:.4f}%")


if __name__ == "__main__":
    chat_app("alpindale/Qwen2.5-0.2B", "alpindale/Qwen2.5-0.2B")

    # train_model(data_path="data/raw/test.parquet")
