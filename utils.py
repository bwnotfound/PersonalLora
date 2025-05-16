from dataclasses import dataclass, field
from typing import Union, Optional, List, Dict, Any
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import evaluate
import numpy as np
import torch


@dataclass
class DataCollatorForSessionLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 2048
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(features, dict):
            input_ids = features["input_ids"]
            labels = features["labels"]
        else:
            input_ids = [_["input_ids"][: self.max_length] for _ in features]
            labels = [_["labels"][: self.max_length] for _ in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            padding_side="left",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            padding_side="left",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )["input_ids"]
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch


@dataclass
class DataCollatorForMovielensTextLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 2048
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(features, dict):
            input_ids = features["input_ids"]
            labels = features["labels"]
            user_indices = features["user_indices"]
            item_indices = features["item_indices"]
        else:
            input_ids = [_["input_ids"][-self.max_length :] for _ in features]
            labels = [_["labels"][-self.max_length :] for _ in features]
            user_indices = [_["user_indices"] for _ in features]
            item_indices = [_["item_indices"] for _ in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )["input_ids"]
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        batch["user_indices"] = torch.tensor(user_indices, dtype=torch.long)
        batch["item_indices"] = torch.tensor(item_indices, dtype=torch.long)

        return batch


@dataclass
class CLMBleu:
    tokenizer: PreTrainedTokenizerBase
    logits: bool = field(default=True)

    def __post_init__(self):
        self.metric = evaluate.load("data/sacrebleu.py")

    def compute_metrics(self, eval_pred):
        preds, refs = eval_pred
        if self.logits:
            preds = np.argmax(preds, axis=-1)
            preds[refs == -100] = self.tokenizer.eos_token_id
        preds[preds == self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
        refs[refs == -100] = self.tokenizer.eos_token_id
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(refs, skip_special_tokens=True)
        return self.metric.compute(predictions=preds, references=refs)


@dataclass
class MovieLensAccuracy:
    tokenizer: PreTrainedTokenizerBase

    def compute_metrics(self, eval_pred, compute_result=None):
        preds, refs = eval_pred.predictions, eval_pred.label_ids
        if len(preds.shape) != len(refs.shape):
            logits_preds = preds
            preds = preds.argmax(-1)
            # preds[refs == -100] = self.tokenizer.eos_token_id
        assert preds.shape == refs.shape, f"{preds.shape} != {refs.shape}"
        preds, refs = preds[:, :-1], refs[:, 1:]
        # preds[preds == self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
        # refs[refs == -100] = self.tokenizer.eos_token_id
        # preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # refs = self.tokenizer.batch_decode(refs, skip_special_tokens=True)
        # pattern = r"<answer>([0-5]+)"
        acc, total = 0, 0
        # for pred, ref in zip(preds, refs):
        # assert ref in [str(x) for x in range(6)]
        # m = re.match(pattern, ref)
        # if m is None:
        #     raise ValueError(f"Invalid reference format: {ref}")
        # ref = m.group(1)
        # m = re.match(pattern, pred)
        # if m is not None:
        #     pred = m.group(1)
        #     if pred == ref:
        #         acc += 1
        acc += ((preds == refs) * (refs != -100)).sum().item()
        total += preds.shape[0]
        return {"accuracy": acc / total if total > 0 else 0}


class RowFormat:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_row(self, instruction, conversations):
        text_list = [f"system: {instruction}"]

        for conversation in conversations:
            assert len(conversation["answer"]) > 0

            text_list.append(f"user: {conversation['question']}")
            text_list.append(f"assistant: {conversation['answer']}")

        new_text_list = []
        not_label_text = []
        for text in text_list:
            if not text.startswith("assistant"):
                not_label_text.append(text)
            else:
                not_label_text.append("assistant: ")
                not_label_text = "\n".join(not_label_text)
                text = text[len("assistant: ") :]
                new_text_list.append((0, not_label_text))
                new_text_list.append((1, text + self.tokenizer.eos_token))
                not_label_text = []

        text_ids_list = []
        labels = []
        for label, text in new_text_list:
            text_ids = self.tokenizer(
                text, truncation=True, max_length=self.max_length
            )["input_ids"]
            text_ids_list.extend(text_ids)
            if label == 0:
                labels.extend([-100] * len(text_ids))
            else:
                labels.extend(text_ids)
        return text_ids_list, labels

    def __call__(self, examples):
        instruction = examples["instruction"]
        conversations = examples["conversations"]
        input_ids = []
        labels = []
        for instruction, conversation in zip(instruction, conversations):
            try:
                text_ids_list, label = self.process_row(instruction, conversation)
            except:
                continue
            input_ids.append(text_ids_list)
            labels.append(label)
        return {"input_ids": input_ids, "labels": labels}


class MovielensTextFormat:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        prompts = examples["prompt"]
        responses = examples["response"]
        input_ids = []
        labels = []
        prompt_ids_list = self.tokenizer(
            prompts, truncation=True, max_length=self.max_length
        )["input_ids"]
        response_ids_list = self.tokenizer(
            responses, truncation=True, max_length=self.max_length
        )["input_ids"]
        for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list):
            if len(prompt_ids) + len(response_ids) > self.max_length:
                raise ValueError(
                    f"Prompt and response combined length exceeds max_length: {len(prompt_ids) + len(response_ids)} > {self.max_length}"
                )
            input_ids.append(prompt_ids + response_ids)
            labels.append([-100] * len(prompt_ids) + response_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "user_indices": [int(x) for x in examples["user_index"]],
            "item_indices": [int(x) for x in examples["item_index"]],
        }


def compute_loss_func(model_output, labels, **kwargs):
    shift_labels = True
    logits = (
        model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    )
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    return loss


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids
