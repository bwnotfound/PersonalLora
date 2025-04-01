import torch.nn as nn
from accelerate import Accelerator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def prepare_inputs_for_generation(self):
        pass


net = Net()

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["linear1", "linear2"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

from module.lora import LoraLinear
import torch.nn as nn

custom_module_mapping = {nn.Linear: LoraLinear}
peft_config._register_custom_module(custom_module_mapping)

net = get_peft_model(net, peft_config)

accelerator = Accelerator()
new_net = accelerator.prepare(net)
for name, param in new_net.named_modules():
    param.bw = "test"

for name, param in net.named_modules():
    print(name)
    if hasattr(param, "bw"):
        print(param.bw)
    else:
        print("no bw")
exit()

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# print(tokenizer.eos_token)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

from module.lora import LoraLinear
import torch.nn as nn

custom_module_mapping = {nn.Linear: LoraLinear}
peft_config._register_custom_module(custom_module_mapping)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


device = "cuda"
model.to(device)

input_text = (
    "user: 介绍你自己并用通俗易懂的语言介绍斐波拉契数列背后的故事。\nassistant: "
)
input_ids = tokenizer(input_text, return_tensors="pt").to(device)
# Generate text
output = model.generate(
    **input_ids,
    max_new_tokens=256,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)
# Decode the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
