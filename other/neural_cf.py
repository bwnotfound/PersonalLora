import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 1. 数据准备
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/raw/ml-1m/ratings.dat", sep="::", names=column_names, engine="python")
user_enc = LabelEncoder(); item_enc = LabelEncoder()
df["user"] = user_enc.fit_transform(df["user_id"])
df["item"] = item_enc.fit_transform(df["item_id"])
n_users, n_items = df["user"].nunique(), df["item"].nunique()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class RatingDataset(Dataset):
    def __init__(self, df):
        self.u = df["user"].values
        self.i = df["item"].values
        self.r = df["rating"].values.astype(np.float32)
    def __len__(self):
        return len(self.u)
    def __getitem__(self, idx):
        return {
            "user": torch.tensor(self.u[idx], dtype=torch.long),
            "item": torch.tensor(self.i[idx], dtype=torch.long),
            "rating": torch.tensor(self.r[idx], dtype=torch.float),
        }

train_loader = DataLoader(RatingDataset(train_df), batch_size=256, shuffle=True)
val_loader   = DataLoader(RatingDataset(val_df),   batch_size=256, shuffle=False)

# 2. NeuMF 模型定义
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, 
                 mf_dim=32, mlp_emb_dim=32, mlp_hidden=[64,32]):
        super().__init__()
        # GMF 部分 Embedding
        self.mf_user_emb = nn.Embedding(n_users, mf_dim)
        self.mf_item_emb = nn.Embedding(n_items, mf_dim)
        # MLP 部分 Embedding
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)
        # MLP 多层感知机
        mlp_layers = []
        input_size = mlp_emb_dim * 2
        for h in mlp_hidden:
            mlp_layers += [nn.Linear(input_size, h), nn.ReLU()]
            input_size = h
        self.mlp = nn.Sequential(*mlp_layers)
        # 最终预测层：GMF 输出 + MLP 输出 拼接后全连接
        predict_size = mf_dim + mlp_hidden[-1]
        self.predict_layer = nn.Linear(predict_size, 1)
        
    def forward(self, user, item):
        # GMF 分支
        mf_u = self.mf_user_emb(user)
        mf_i = self.mf_item_emb(item)
        mf_out = mf_u * mf_i           # 元素乘
        
        # MLP 分支
        mlp_u = self.mlp_user_emb(user)
        mlp_i = self.mlp_item_emb(item)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_in)
        
        # 拼接并预测
        x = torch.cat([mf_out, mlp_out], dim=-1)
        return self.predict_layer(x).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuMF(n_users, n_items).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 3. 训练与验证
def rmse(preds, targets):
    return torch.sqrt(((preds - targets)**2).mean())

rating_min, rating_max = 1, 5
for epoch in range(1, 11):
    # 训练
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        u, i, r = batch["user"].to(device), batch["item"].to(device), batch["rating"].to(device)
        pred = model(u, i)
        loss = criterion(pred, r)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # 验证
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validating"):
            u, i, r = batch["user"].to(device), batch["item"].to(device), batch["rating"].to(device)
            p = model(u, i)
            all_pred.append(p.cpu()); all_true.append(r.cpu())
    all_pred = torch.cat(all_pred).numpy(); all_true = torch.cat(all_true).numpy()
    # RMSE
    val_rmse = rmse(torch.tensor(all_pred), torch.tensor(all_true))
    # 精度（分类准确率）
    y_pred = np.clip(np.rint(all_pred), rating_min, rating_max).astype(int)
    y_true = all_true.astype(int)
    accuracy = (y_pred == y_true).mean()
    print(f"Epoch {epoch:02d} - Val RMSE: {val_rmse:.4f}, Acc: {accuracy:.4f}")

# 4. 单例预测
sample = val_df.iloc[0]
u_idx = torch.tensor(sample["user"]).to(device)
i_idx = torch.tensor(sample["item"]).to(device)
model.eval()
from bwtools.log import TimeCounter
with torch.no_grad():
    pr = model(u_idx.unsqueeze(0), i_idx.unsqueeze(0)).item()
    with TimeCounter():
        pr = model(u_idx.unsqueeze(0), i_idx.unsqueeze(0)).item()
print(f"真实: {sample['rating']:.1f}, 预测: {pr:.4f}")
