import os
import numpy as np
import pickle
import time
import math

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix

from tools import TimeCounter, AvgRecorder


class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.users = data["users"]
        self.movies = data["movies"]
        self.ratings = data["ratings"]
        with TimeCounter("shuffle ratings"):
            np.random.shuffle(self.ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.ratings[idx]
        gender, age, occupation = self.users[user_id]
        genres = self.movies[movie_id]
        return user_id, gender, age, occupation, movie_id, genres, rating - 1

    def collate_fn(self, batch):
        user_data = torch.tensor([x[:4] for x in batch], dtype=torch.long)
        movie_id = torch.tensor([x[4] for x in batch], dtype=torch.long)
        genres_shape = [len(x[5]) for x in batch]
        max_len = max(genres_shape)
        genres_shape = torch.tensor(genres_shape, dtype=torch.long)
        genres = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            genres[i, : len(x[5])] = torch.tensor(x[5], dtype=torch.long)
        rating = torch.tensor([x[6] for x in batch], dtype=torch.long)
        return user_data, movie_id, genres, genres_shape, rating


class MLP(nn.Module):
    def __init__(self, mlp_dims, activation=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(mlp_dims) - 1):
            self.layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i != len(mlp_dims) - 2:
                self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UserEmbedding(nn.Module):
    def __init__(self, num_users, embed_dim):
        super().__init__()
        self.user_id_embed = nn.Embedding(num_users + 1, embed_dim)
        self.gender_embed = nn.Embedding(2 + 1, embed_dim)
        self.age_embed = nn.Embedding(7 + 1, embed_dim)
        self.occupation_embed = nn.Embedding(21 + 1, embed_dim)
        self.post_layer = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, user_data: torch.Tensor):
        assert user_data.shape[1] == 4 and len(user_data.shape) == 2
        user_id, gender, age, occupation = (
            user_data[:, 0],
            user_data[:, 1],
            user_data[:, 2],
            user_data[:, 3],
        )
        user_id_embed = self.user_id_embed(user_id)
        age_embed = self.age_embed(age)
        occupation_embed = self.occupation_embed(occupation)
        gender_embed = self.gender_embed(gender)
        x = torch.cat([user_id_embed, gender_embed, age_embed, occupation_embed], dim=1)
        x = self.post_layer(x)
        return x


class MovieEmbedding(nn.Module):
    def __init__(self, num_movies, embed_dim, num_layers=1):
        super().__init__()
        self.movie_id_embed = nn.Embedding(num_movies + 1, embed_dim)
        self.genres_embed = nn.Embedding(18 + 1, embed_dim)
        self.genres_rnn = nn.LSTM(
            embed_dim, embed_dim, num_layers=num_layers, batch_first=True
        )
        self.genres_embed_post_l = nn.Linear(embed_dim, embed_dim)
        self.post_layer = nn.Linear(2 * embed_dim, embed_dim)

    def forward(
        self,
        movie_id: torch.Tensor,
        genres: torch.Tensor,
        genres_shape: torch.Tensor,
    ):
        movie_id_embed = self.movie_id_embed(movie_id)
        genres_embed = self.genres_embed(genres)
        assert torch.min(genres_shape).item() > 0
        genres_embed = pack_padded_sequence(
            genres_embed,
            genres_shape.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        genres_embed, _ = self.genres_rnn(genres_embed)
        genres_embed, _ = pad_packed_sequence(
            genres_embed,
            batch_first=True,
        )
        genres_embed = genres_embed[
            torch.arange(genres_embed.size(0)), genres_shape - 1
        ]
        genres_embed = self.genres_embed_post_l(genres_embed)
        x = torch.cat([movie_id_embed, genres_embed], dim=1)
        x = self.post_layer(x)
        return x


class DLRM(nn.Module):
    def __init__(self, user_embed, movie_embed, num_classes, embed_dim, mlp_dims):
        super().__init__()

        self.user_embed = user_embed
        self.movie_embed = movie_embed

        self.mlp = MLP([2 * embed_dim] + mlp_dims)
        self.fc = nn.Linear(mlp_dims[-1], num_classes)

    def forward(
        self,
        user_data: torch.Tensor,
        movie_id: torch.Tensor,
        genres: torch.Tensor,
        genres_shape: torch.Tensor,
    ):
        user_embed = self.user_embed(user_data)
        movie_embed = self.movie_embed(movie_id, genres, genres_shape)
        x = torch.cat([user_embed, movie_embed], dim=1)
        x = self.mlp(x)
        x = self.fc(x)
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

    def train_full(self):
        self.user_embed.train()
        self.movie_embed.train()
        self.mlp.train()
        self.fc.train()

    def train_emb(self):
        self.user_embed.train()
        self.movie_embed.train()
        self.mlp.eval()
        self.fc.eval()

    def train_mlp(self):
        self.user_embed.eval()
        self.movie_embed.eval()
        self.mlp.train()
        self.fc.train()

    def train(self):
        self.train_full()

    def eval(self):
        self.user_embed.eval()
        self.movie_embed.eval()
        self.mlp.eval()
        self.fc.eval()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    batch_size = 256
    eval_batch_size = 256
    epochs = 100
    is_classification = True

    train_dataset = MovieLensDataset("data/train/movielens.pkl")
    eval_dataset = MovieLensDataset("data/eval/movielens.pkl")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn,
    )
    model = DLRM(
        UserEmbedding(6040 + 1, 64),
        MovieEmbedding(3952 + 1, 64),
        5 if is_classification else 1,
        64,
        [128, 128],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if is_classification:
        loss_func = nn.CrossEntropyLoss()
    else:

        def loss_func(x, y):
            return nn.MSELoss()(x, y.float())

    model.to(device)
    loss_avg_record = AvgRecorder(20)
    for epoch in range(epochs):
        model.train()
        t_bar = tqdm(total=len(train_loader))
        last_time = 0
        for data in train_loader:
            user_data, movie_id, genres, genres_shape, rating = [
                x.to(device) for x in data
            ]
            output = model(user_data, movie_id, genres, genres_shape)
            loss = loss_func(output, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg_record(loss.item())
            if time.time() - last_time > 0.5:
                t_bar.set_postfix_str(f"loss: {loss_avg_record}")
                last_time = time.time()
            t_bar.update()
        t_bar.close()
        model.eval()
        labels, preds = [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader, leave=False)):
                user_data, movie_id, genres, genres_shape, rating = [
                    x.to(device) for x in data
                ]
                output = model(user_data, movie_id, genres, genres_shape)
                if is_classification:
                    predicted = torch.argmax(output, dim=1)
                else:
                    predicted = output.round()
                for j in range(rating.size(0)):
                    labels.append(rating[j].item())
                    preds.append(predicted[j].item())
        # if is_classification:
        cm = confusion_matrix(labels, preds)
        acc = cm.diagonal().sum() / cm.sum()
        tqdm.write(f"Epoch {epoch}: acc: {acc}, cm: {cm}")
        # else:
        #     labels, preds = np.array(labels), np.array(preds)
        #     mse = ((labels - preds) ** 2).mean()
        #     tqdm.write(
        #         f"Epoch {epoch}: mse: {mse}, l1: {np.abs(labels - preds).mean()}"
        #     )


def test_dataset_rating_distribution():
    dataset = MovieLensDataset("data/eval/movielens.pkl")
    ratings = [x[2] for x in dataset.ratings]
    ratings_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for rating in ratings:
        ratings_dict[rating] = ratings_dict.get(rating, 0) + 1
    ratings_dict = {
        k: (f"{(v / len(ratings)) * 100:2f}%", v) for k, v in ratings_dict.items()
    }
    print(ratings_dict)


if __name__ == "__main__":
    main()
    # test_dataset_rating_distribution()
