# 复现步骤

## 项目结构

1. `module/lora.py`是核心动态LoRA实现，以peft风格。包括论文中提到的根据用户、电影动态加载LoRA_B矩阵，目前router暂时按照等权重加和实现。
2. `train.sh`是训练shell脚本，`train.py`则是主要训练脚本
3. `gen_data.py`是数据预处理脚本，包括论文中的文本格式化、额外信息附加
4. `webui.py`是可视化脚本，目前还未完善

## 运行环境
1. 安装依赖库
```shell
# python版本: 3.10
pip install -r requirements.txt
# 其中torch请自行安装cuda版本
```
2. [wandb](https://wandb.ai/site)注册账号，并获取[api key](https://wandb.ai/authorize)。
3. 数据选择[MovieLens-1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip)，获取数据集后将文件夹放置在`data/raw`下并命名为`ml-1m`。启动`gen_data.py`即可在`data/train/movielens_llm`获取训练测试集
4. 训练脚本启动`train.sh`，修改其中参数，使其对应本地大语言模型`Qwen2.5-0.5B-Instruct`路径、输出路径、训练和测试集路径
5. 训练参数目前没有提供shell参数控制，如需修改请更改`train.py`的`train_model`函数中`config`定义
6. 理论上不需要`eval.py`进行测试，因为测试在训练过程中进行

