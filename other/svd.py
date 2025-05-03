# -*- coding: utf-8 -*-
"""
实现思路：
1. 使用 Surprise 库加载 MovieLens 1M 数据集；
2. 使用 SVD 算法进行训练与交叉验证；
3. 计算测试集上的 RMSE 评估指标；
4. 对预测结果四舍五入后与真实评分做对比，输出准确率；
5. 演示对单个用户-物品对的预测。
"""

'''
准确率 44.77%
'''

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split
import numpy as np

# 本地数据路径
data_path = "data/raw/ml-1m/ratings.dat"

# 1. 定义 Reader —— 根据文件格式指定分隔符、字段顺序和评分范围
reader = Reader(line_format="user item rating timestamp", sep="::", rating_scale=(1, 5))

# 2. 从文件加载数据
data = Dataset.load_from_file(data_path, reader=reader)

# 3. 交叉验证评估
algo = SVD(n_factors=50, lr_all=0.005, reg_all=0.02)
cv_results = cross_validate(algo, data, measures=["RMSE"], cv=5, verbose=True)

# 4. 划分训练/测试集并训练
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
algo.fit(trainset)

# 5. 在测试集上评估
predictions = algo.test(testset)

# 5.1 计算 RMSE
rmse = accuracy.rmse(predictions)

# 5.2 计算准确率（Exact match of rounded prediction）
# 提取真实评分与预测评分
y_true = np.array([pred.r_ui for pred in predictions], dtype=int)
# 四舍五入并截断到 1–5 之间
y_pred = np.array([int(round(pred.est)) for pred in predictions])
y_pred = np.clip(y_pred, 1, 5)
# 计算准确率
accuracy_score = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy_score:.4f}")

# 6. 单例预测示例
uid, iid = str(196), str(302)  # 用户 ID 和 电影 ID 必须是字符串
pred = algo.predict(uid, iid, r_ui=4.0, verbose=True)
