# data analyse to train/test/attribute.csv
import pandas as pd
import matplotlib.pyplot as plt

data ='D:/machine-learning/Recommend-System/data/train.csv'

df = pd.read_csv(data)

# 计算每个用户的平均评分
user_mean_scores = df.groupby('user_id')['score'].mean()
print("每个用户的平均评分：")
print(user_mean_scores)

# 计算每个物品的平均评分
item_mean_scores = df.groupby('item_id')['score'].mean()
print("\n每个物品的平均评分：")
print(item_mean_scores)

# 计算方差
user_variance = df.groupby('user_id')['score'].var()
print("\n每个用户评分的方差：")
print(user_variance)

# 因为有些物品只有一次评分或者没有评分，导致方差为NaN
# 在进行推荐的时候可以考虑把这些物品去掉，因为参考性太差了
item_variance = df.groupby('item_id')['score'].var()
item_variance = item_variance[item_variance.notnull()]
print("\n每个物品评分的方差：")
print(item_variance)

# item0 = df[df['item_id'] == 0] # 获得item_id为 0 的所有项
# print(item0)
#         user_id  item_id  score
# 665423     2435        0     50

#---------------------图表绘制--------------------------#
import matplotlib.pyplot as plt

# ----------统计每个评分对应的 user 数量------------------*
score_counts = df.groupby('score')['user_id'].count()

# 创建图表
plt.figure(figsize=(10,6))

# 绘制柱状图
plt.bar(score_counts.index, score_counts.values)

# 设置图表标题和标签
plt.title("The Number of Users for each Rating")
plt.xlabel("Rating")
plt.ylabel("User_num")

# 设置纵轴范围
plt.ylim(0, score_counts.max() + 50000)

# 显示图表
plt.savefig('D:/machine-learning/Recommend-System/image/score_counts.jpg')
plt.show()

# --------------------计算每个用户的平均评分--------------------- #
AvgScoreToUser = df.groupby('user_id')['score'].mean()
print()

plt.figure(figsize=(10,6))

plt.hist(AvgScoreToUser, bins=100, range=(0,100), edgecolor='black')

plt.title("Average Rating per User")
plt.xlabel("Rating")
plt.ylabel("User_num")

plt.savefig('D:/machine-learning/Recommend-System/image/AvgScoreToUser.jpg')
plt.show()

# ------------------- 计算每个物品的平均得分---------------------#

AvgScoreToItem = df.groupby('item_id')['score'].mean()
print()

plt.figure(figsize=(10,6))

plt.hist(AvgScoreToItem, bins=100, range=(0,100), edgecolor='black')

plt.title("Average score per Item")
plt.xlabel("Score")
plt.ylabel("Item_num")

plt.savefig('D:/machine-learning/Recommend-System/image/AvgScoreToItem.jpg')
plt.show()