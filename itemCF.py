# realize item-item collaborative filtering:
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix

# mission1:calculate the similarity between objects:
trainPath = 'D:/machine-learning/Recommend-System/data/train.csv'
train_data = pd.read_csv(trainPath)

def compute_item_similarity(train_data):
    # Step 1: 读取数据，转到函数外进行，因为后面 get_user_favorites 也要用 train_data
    # Step 2: 计算物品的平均评分
    item_mean = train_data.groupby('item_id')['score'].mean()
    
    # Step 3: 建立用户对物品评分的字典
    user_item_ratings = defaultdict(dict)
    for row in train_data.itertuples(index=False):
        user_item_ratings[row.user_id][row.item_id] = row.score
    
    # 获取所有物品的列表
    items = list(item_mean.index)
    num_items = len(items)
    print(num_items)
    # Step 4: 计算物品之间的相似度
    similarity_matrix = lil_matrix((num_items, num_items))

    # 如果这里用 GPU 计算 直接爆表，每个int 8 bytes =32b，总共 455691 * 455691 * 8 bytes
    # similarity_matrix = cp.zeros((num_items, num_items))
    
    for i in range(num_items):
        for j in range(i, num_items):
            item_i = items[i]
            item_j = items[j]
            
            # 获取共同评价过 item_i 和 item_j 的用户
            users_i = {user for user, ratings in user_item_ratings.items() if item_i in ratings}
            users_j = {user for user, ratings in user_item_ratings.items() if item_j in ratings}
            common_users = users_i.intersection(users_j)
            if len(common_users) == 0:
                similarity = 0
            else:
                sum_ij = 0
                sum_i = 0
                sum_j = 0
                
                for user in common_users:
                    r_ui = user_item_ratings[user][item_i]
                    r_uj = user_item_ratings[user][item_j]
                    
                    r_i_bar = item_mean[item_i]
                    r_j_bar = item_mean[item_j]

                    print(r_ui,r_uj,r_i_bar,r_j_bar)  # 这回没问题了
                    sum_ij += (r_ui - r_i_bar) * (r_uj - r_j_bar)
                    sum_i += (r_ui - r_i_bar) ** 2
                    sum_j += (r_uj - r_j_bar) ** 2
                
                if sum_i == 0 or sum_j == 0:
                    similarity = 0
                else:
                    similarity = sum_ij / (np.sqrt(sum_i) * np.sqrt(sum_j))
                    print(similarity)
            if similarity != 0:
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # 相似度矩阵对称
    # 因为 item0 只有一个评分为 50，导致所有的 r_ui - r_i_bar=0，产出的所有 similarity=0
    # 应该筛选一下，存在方差的 item 之间再计算，不过差别也不大

    # 将稀疏矩阵转换为 DataFrame 对象
    df_similarity = pd.DataFrame(similarity_matrix.toarray())

    # 将DataFrame对象写入CSV文件
    df_similarity.to_csv('similarity_matrix.csv', index=False, header=False)

    return similarity_matrix

# similarity_matrix = compute_item_similarity(trainPath)

# mission2 : create the recommend list
# 有了物品之间的相似度后，需要实现test中用户给物品打分：
# user_id = 0,item_id = 127640 求 score
# 先获得 user_id 喜欢的书籍 ，个数为 min(打分书本个数，K 本)
# 再按照 所写流程 计算 预测评分


testPath = 'D:/machine-learning/Recommend-System/data/test.csv'

def get_user_favorites(user_id, K=5):
    # return list[ [item1,score1],[item2,score2], ... ]

    # 获取用户评分过的所有items，在 train_data 中
    user_ratings = train_data[train_data.user_id == user_id]
    
    # 根据评分对这些items进行排序
    sorted_items = user_ratings.sort_values(by='score', ascending=False)
    
    # 返回前K个items，如果不够 K 个就返回全部items
    if len(sorted_items) >= K:
        top_k_items = sorted_items[:K]
    else:
        top_k_items = sorted_items
    return top_k_items[['item_id', 'score']].values.tolist()

# L = get_user_favorites(0)
# print(L)

# 根据相似度计算预测评分
def predict_score(user_id, item_id, similarity_matrix, K=5):
    favorites = get_user_favorites(user_id, K)
    
    FinalPred = 0.0
    
    for fav_item,score in favorites:
        if fav_item in similarity_matrix.index and item_id in similarity_matrix.columns:
            # 获取相似度
            sim = similarity_matrix.at[fav_item, item_id]
            FinalPred += score * sim
    
    return FinalPred

def main():
    # 计算物品间的相似度矩阵
    similarity_matrix = compute_item_similarity(train_data)
    test_data = pd.read_csv(testPath)
    # 预测评分
    predictions = []
    for _, row in test_data.iterrows():
        # 返回 行索引 和 行数据，这里使用 行数据
        user_id = row['user_id']
        item_id = row['item_id']
        predicted_score = predict_score(user_id, item_id, similarity_matrix)
        predictions.append(predicted_score)
    
    # 所有行都计算完成后，将预测结果写入test.csv
    test_data['score'] = predictions
    test_data.to_csv('test_predictions.csv', index=False)

    # 最后再按要求，把 test_predictions.csv 转为对应的 txt 文件即可
    
if __name__ == "__main__":
    main()