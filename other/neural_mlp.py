# -*- coding: utf-8 -*-
"""
实现思路：
1. 用 pandas 读取 MovieLens 100k 原始文件；
2. 使用 LabelEncoder 将用户 ID 和物品 ID 转换为连续索引；
3. 构建 PyTorch Dataset & DataLoader；
4. 定义 NeuralCF 模型：用户与物品嵌入 + MLP；
5. 训练模型并在验证集上评估 RMSE；
6. 演示对单个样本的预测。
"""


'''
准确率 40.00%
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from bwtools.log import NumberRecoder

from sklearn.metrics import confusion_matrix

# 1. 读取原始评分数据
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(
    "data/raw/ml-1m/ratings.dat", sep="::", names=column_names
)  # MovieLens 100k 原始文件路径 :contentReference[oaicite:7]{index=7}

# 2. 编码用户与物品 ID
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user"] = user_enc.fit_transform(df["user_id"])
df["item"] = item_enc.fit_transform(df["item_id"])
n_users, n_items = df["user"].nunique(), df["item"].nunique()

# 3. 划分训练/验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = df["user"].values
        self.items = df["item"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "item": torch.tensor(self.items[idx], dtype=torch.long),
            "rating": torch.tensor(self.ratings[idx], dtype=torch.float),
        }


train_loader = DataLoader(RatingDataset(train_df), batch_size=256, shuffle=True)
val_loader = DataLoader(RatingDataset(val_df), batch_size=256, shuffle=False)


# 4. 定义 NeuralCF 模型
class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, hidden_dims=[64, 32]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        layers = []
        input_dim = emb_dim * 2
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, item):
        u = self.user_emb(user)
        v = self.item_emb(item)
        x = torch.cat([u, v], dim=-1)
        return self.mlp(x).squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralCF(n_users, n_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


# 5. 训练与验证
def rmse(preds, targets):
    return torch.sqrt(((preds - targets) ** 2).mean())


for epoch in range(1, 11):
    model.train()
    t_bar = tqdm(train_loader)
    loss_reocder = NumberRecoder()
    for batch in t_bar:
        u = batch["user"].to(device)
        i = batch["item"].to(device)
        r = batch["rating"].to(device)
        pred = model(u, i)
        loss = criterion(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_reocder.update(loss.item())
        t_bar.set_description(f"Epoch {epoch:02d} - Loss: {loss_reocder.average:.4f}")
    t_bar.close()
    # 验证
    model.eval()
    with torch.no_grad():
        all_pred, all_r = [], []
        for batch in tqdm(val_loader):
            u = batch["user"].to(device)
            i = batch["item"].to(device)
            r = batch["rating"].to(device)
            p = model(u, i)
            all_pred.append(p.cpu())
            all_r.append(r.cpu())
        all_pred = torch.cat(all_pred)
        all_r = torch.cat(all_r)
        # confusion = confusion_matrix(all_r.numpy(), all_pred.numpy())
        # acc = confusion.diagonal().sum() / confusion.sum()
        # 将连续预测值转换为离散类别
        y_pred = np.rint(all_pred.numpy()).astype(int)
        y_pred = np.clip(y_pred, 1, 5)
        y_true = all_r.numpy().astype(int)

        # 计算混淆矩阵
        confusion = confusion_matrix(y_true, y_pred, labels=list(range(1, 5 + 1)))
        acc = np.trace(confusion) / confusion.sum()
        print(
            f"Epoch {epoch:02d} - Acc: {acc*100:.2f}% - Confusion Matrix:\n{confusion}"
        )
        val_rmse = rmse(all_pred, all_r)
    print(
        f"Epoch {epoch:02d} - Train Loss: {loss_reocder.average:.4f}, Val RMSE: {val_rmse:.4f}"
    )

# 6. 单例预测示例
# 选取 DataFrame 中第一个样本
sample = val_df.iloc[0]
u_idx = torch.tensor(sample["user"]).to(device)
i_idx = torch.tensor(sample["item"]).to(device)
model.eval()
with torch.no_grad():
    pred_rating = model(u_idx.unsqueeze(0), i_idx.unsqueeze(0)).item()
print(f"真实评分: {sample['rating']:.1f}, 预测评分: {pred_rating:.4f}")
