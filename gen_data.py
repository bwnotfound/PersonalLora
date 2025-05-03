import random
import os
import pickle
import json

import pandas as pd
from tqdm import tqdm


def gen_random_calc_task():

    df = []

    for i in range(50000):
        instruction = "Solve the equation"
        loop = random.randint(1, 9)
        conversation = []
        for j in range(loop):
            x, y = random.randint(1, 1000), random.randint(1, 1000)
            mode = random.randint(0, 3)
            if mode == 0:
                z = x * y
                question = f"{x} * {y} = ?"
                answer = f"{z}"
                conversation.append({"question": question, "answer": answer})
            if mode == 1:
                z = x - y
                question = f"{x} - {y} = ?"
                answer = f"{z}"
                conversation.append({"question": question, "answer": answer})
            if mode == 2:
                z = x / y
                question = f"{x} / {y} = ?"
                answer = f"{z:.2f}"
                conversation.append({"question": question, "answer": answer})
            if mode == 3:
                z = x + y
                question = f"{x} + {y} = ?"
                answer = f"{z}"
                conversation.append({"question": question, "answer": answer})
        df.append({"instruction": instruction, "conversations": conversation})
    df = pd.DataFrame(df)
    print(df.head())
    eval_ratio = 0.1
    eval_df = df.sample(frac=eval_ratio)
    train_df = df.drop(eval_df.index)
    train_df.to_parquet("data/raw/simple_test/train.parquet")
    eval_df.to_parquet("data/raw/simple_test/eval.parquet")
    # df.to_parquet("data/raw/simple_test/test.parquet")


def form_sft_from_minimind():
    raw_path = "data/raw/sft_mini_512.jsonl"
    import jsonlines

    with jsonlines.open(raw_path) as reader:
        data = [item for item in reader]
    # print(data[0])
    df = []
    for item in tqdm(data):
        conversation = []
        assert len(item["conversations"]) % 2 == 0
        for i in range(0, len(item["conversations"]), 2):
            assert item["conversations"][i]["role"] == "user"
            assert item["conversations"][i + 1]["role"] == "assistant"
            question = item["conversations"][i]["content"]
            answer = item["conversations"][i + 1]["content"]
            conversation.append({"question": question, "answer": answer})
        df.append({"instruction": "", "conversations": conversation})
    df = pd.DataFrame(df)
    print(df.head())
    eval_ratio = 0.05
    eval_df = df.sample(frac=eval_ratio)
    train_df = df.drop(eval_df.index)
    train_df.to_parquet("data/raw/sft_mini/train.parquet")
    eval_df.to_parquet("data/raw/sft_mini/eval.parquet")


def gen_movielens_dataset():
    """
    movie的genres需要按照字典序排序，否则处理数据时由于先后顺序会有问题，即字典需要是有序的
    """

    eval_ratio = 0.1
    data_dir = "data/raw/ml-1m"
    suffix = ".dat"
    # suffix = ".csv"
    users_path = os.path.join(data_dir, f"users{suffix}")
    movies_path = os.path.join(data_dir, f"movies{suffix}")
    ratings_path = os.path.join(data_dir, f"ratings{suffix}")
    data_name = os.path.basename(data_dir)
    with open(users_path, "r", encoding="utf-8") as f:
        users = f.readlines()
    new_users = {}
    user_age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    for user_str in tqdm(users, leave=False):
        """
        UserID::Gender::Age::Occupation::Zip-code
        1::F::1::10::48067
        1081::M::18::4::68144-2410
        """
        user = user_str.strip().split("::")
        user[1] = 0 if user[1] == "M" else 1
        user = user[:-1]
        user = [int(x) for x in user]
        user[2] = user_age_map[user[2]]
        new_users[user[0]] = user[1:]
    users = new_users
    with open(movies_path, "r", encoding="utf-8") as f:
        movies = f.readlines()
    new_movies = {}
    movie_geners_map = {
        "Action": 0,
        "Adventure": 1,
        "Animation": 2,
        "Children's": 3,
        "Comedy": 4,
        "Crime": 5,
        "Documentary": 6,
        "Drama": 7,
        "Fantasy": 8,
        "Film-Noir": 9,
        "Horror": 10,
        "Musical": 11,
        "Mystery": 12,
        "Romance": 13,
        "Sci-Fi": 14,
        "Thriller": 15,
        "War": 16,
        "Western": 17,
    }
    for movie_str in tqdm(movies, leave=False):
        """
        MovieID::Title::Genres
        1::Toy Story (1995)::Animation|Children's|Comedy
        3948::Meet the Parents (2000)::Comedy
        * Action
        * Adventure
        * Animation
        * Children's
        * Comedy
        * Crime
        * Documentary
        * Drama
        * Fantasy
        * Film-Noir
        * Horror
        * Musical
        * Mystery
        * Romance
        * Sci-Fi
        * Thriller
        * War
        * Western
        """
        movie = movie_str.strip().split("::")
        movie = [
            int(movie[0]),
            list(map(lambda x: movie_geners_map[x], movie[2].split("|"))),
        ]
        new_movies[movie[0]] = movie[1]
    movies = new_movies
    with open(ratings_path, "r", encoding="utf-8") as f:
        ratings = f.readlines()
    train_ratings, eval_ratings = [], []
    user_ratings_dict = {}
    for rating_str in tqdm(ratings, leave=False):
        """
        UserID::MovieID::Rating::Timestamp
        1::1193::5::978300760
        1::661::3::978302109
        """
        rating = rating_str.strip().split("::")
        rating = [int(x) for x in rating]
        user_ratings_dict.setdefault(rating[0], []).append(
            (rating[1], rating[2], rating[3])
        )
    for user_id in user_ratings_dict.keys():
        ratings = user_ratings_dict[user_id]
        ratings.sort(key=lambda x: x[2])
        ratings = [(x[0], x[1]) for x in ratings]
        train_ratings.extend(
            [
                [user_id, movie_id, rating]
                for movie_id, rating in ratings[: int(len(ratings) * (1 - eval_ratio))]
            ]
        )
        eval_ratings.extend(
            [
                [user_id, movie_id, rating]
                for movie_id, rating in ratings[int(len(ratings) * (1 - eval_ratio)) :]
            ]
        )
    train_data = {"users": users, "movies": movies, "ratings": train_ratings}
    eval_data = {"users": users, "movies": movies, "ratings": eval_ratings}

    print(
        f"train_data: {len(train_data['ratings'])}, eval_data: {len(eval_data['ratings'])}"
    )
    with open(os.path.join("data/train", "movielens.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join("data/eval", "movielens.pkl"), "wb") as f:
        pickle.dump(eval_data, f)


def generate_movielens_llm_dataset():
    """
    生成适用于 LLM 训练的 MovieLens 数据集，并保存为 parquet 文件。
    输出文件包含三列：["user_index", "prompt", "response"]。

    每条记录对应 ratings 文件中的一条评分，构造规则如下：
      - user_index：评分记录中的用户 ID
      - prompt：正确的 JSON 格式字符串，格式示例：
          {"user": {"age": "18-24", "gender": "female", "occupation": "programmer"}, "movie": {"title": "Toy Story (1995)", "genres": "Animation,Children's,Comedy"}}, predict the rating:
      - response：格式为 <answer>{rating}</answer><|im_end|>
    """
    # 文件路径设置
    data_dir = "data/raw/ml-1m"
    file_suffix = ".dat"
    eval_ratio = 0.005

    users_file = os.path.join(data_dir, f"users{file_suffix}")
    movies_file = os.path.join(data_dir, f"movies{file_suffix}")
    ratings_file = os.path.join(data_dir, f"ratings{file_suffix}")

    user_num = -1
    # user_num = 1000

    # 定义 occupation 映射
    occupation_map = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer",
    }

    # 定义 age 映射（实际年龄段）
    age_map = {
        "1": "Under 18",
        "18": "18-24",
        "25": "25-34",
        "35": "35-44",
        "45": "45-49",
        "50": "50-55",
        "56": "56+",
    }

    # 解析用户数据，文件格式：UserID::Gender::Age::Occupation::Zip-code
    users = {}
    with open(users_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 5:
                continue
            user_id = int(parts[0])
            if user_num != -1 and user_id + 1 > user_num:
                continue
            gender_raw = parts[1]
            # gender 保持原始表示，但转换为完整字符串
            gender = "male" if gender_raw == "M" else "female"
            age_code = parts[2]
            age_str = age_map.get(age_code, age_code)
            occupation_code = int(parts[3])
            occupation = occupation_map.get(occupation_code, "other")
            users[user_id] = {
                "age": age_str,
                "gender": gender,
                "occupation": occupation,
            }

    # 解析电影数据，文件格式：MovieID::Title::Genres
    movies = {}
    with open(movies_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id = int(parts[0])
            title = parts[1]
            genres_list = parts[2].split("|")
            # 对 genres 按字典序排序并用逗号连接
            genres_sorted = sorted(genres_list)
            genres_str = ",".join(genres_sorted)
            movies[movie_id] = {"title": title, "genres": genres_str}

    # 构造训练数据，每一条评分记录转换为一行
    data_rows = []
    with open(ratings_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Processing ratings"):
        parts = line.strip().split("::")
        if len(parts) != 4:
            continue
        user_id = int(parts[0])
        movie_id = int(parts[1])
        rating = int(parts[2])

        if user_num != -1 and user_id + 1 > user_num:
            continue

        # 获取用户和电影信息
        user_info = users.get(user_id)
        movie_info = movies.get(movie_id)
        if not user_info or not movie_info:
            continue

        # 构造 prompt：先生成 dict 后转换为 JSON 字符串，确保 key 均用双引号括起
        prompt_dict = {
            "user": {
                "age": user_info["age"],
                "gender": user_info["gender"],
                "occupation": user_info["occupation"],
            },
            "movie": {"title": movie_info["title"], "genres": movie_info["genres"]},
        }
        prompt = json.dumps(prompt_dict, ensure_ascii=False) + ", predict the rating:"

        # 构造 response
        response = f"{rating}"

        data_rows.append(
            {
                "user_index": int(user_id),
                "item_index": int(movie_id),
                "prompt": prompt,
                "response": response,
            }
        )

    # 转换为 DataFrame 并保存为 parquet 文件
    df = pd.DataFrame(
        data_rows, columns=["user_index", "item_index", "prompt", "response"]
    )
    train_df = df.sample(frac=1 - eval_ratio)
    eval_df = df.drop(train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    os.makedirs("data/train/movielens_llm", exist_ok=True)
    train_path = os.path.join("data/train/movielens_llm", "train.parquet")
    eval_path = os.path.join("data/train/movielens_llm", "eval.parquet")
    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    print(f"train_data: {len(train_df)}, eval_data: {len(eval_df)}")
    print(f"train data saved to {train_path}")
    print(f"eval data saved to {eval_path}")


if __name__ == "__main__":
    # gen_random_calc_task()
    # form_sft_from_minimind()
    # gen_movielens_dataset()
    generate_movielens_llm_dataset()
