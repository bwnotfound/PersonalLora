import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

from tools import TimeCounter


class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        if isinstance(data_path, str):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = data_path
        self.users = data["users"]
        self.movies = data["movies"]
        self.ratings = data["ratings"]
        with TimeCounter("shuffle ratings"):
            np.random.shuffle(self.ratings)
        self.user_ratings_dict = {}
        for i, (user_id, movie_id, rating) in enumerate(self.ratings):
            self.user_ratings_dict.setdefault(user_id, []).append((i, movie_id, rating))

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.ratings[idx]
        gender, age, occupation = self.users[user_id]
        genres = self.movies[movie_id]
        return user_id, gender, age, occupation, movie_id, genres, rating - 1, idx

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

    def text_collate_fn(self, batch):
        user_ids = [x[0] for x in batch]
        select_ids = [x[-1] for x in batch]
        for user_id, select_idx in zip(user_ids, select_ids):
            for i in range(len(self.user_ratings_dict[user_id])):
                if self.user_ratings_dict[user_id][i][0] == select_idx:
                    break
            else:
                raise ValueError
            ratings_history = self.user_ratings_dict[user_id][: i + 1]
