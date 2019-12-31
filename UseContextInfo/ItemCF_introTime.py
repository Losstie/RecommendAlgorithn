"""
利用时间上下文信息的ItemCF
"""

from __future__ import division
import pandas as pd
import numpy as np
import math
import time


class ItemCF_introTime(object):
    def __init__(self, alpha=0.3, beta = 0.15,normalized=False):
        """
            @function: 算法参数初始化
            @param:
                alpha   参数范围 [0,1] 时间衰减数，如果系统中用户兴趣变化很快则取较大值，反之需要取较小值。
                beta 参数范围 [0,1] 时间衰减数 修正预测公式
                normalized {Boolean} 是否进行相似矩阵归一化
        """
        self.alpha = alpha
        self.beta = beta
        self.normalized = normalized
    def timeweak(self, t1, t2):
        """
            时间衰减函数
        """
        return 1.0 / (1 + self.alpha * abs(t1 - t2))

    def fit(self, train):
        """
            @function: 模型训练 得到物品相似矩阵W

            @param:
                train {DataFrame} user item rating 隐反馈数据集有历史行为的item rating都为1
        """

        # 获得列名列表
        self.COLUMNS = train.columns
        print(self.COLUMNS)
        # index:itemId list:userIdList
        items_users = train.groupby(self.COLUMNS[1]).agg([list])[self.COLUMNS[0]]
        # 转化为用户-物品-分数字典 {user: {item: rating}}
        self.ratingDict = dict()
        for _, row in train.iterrows():
            user, item, rating, timeStamp = row
            self.ratingDict.setdefault(user,{}).update({item: {'rat': rating, 'time': timeStamp}})
        # 创建物品-用户列表字典 {item: [user1，user2]}
        self.itemUser = dict(zip(items_users.index, items_users.list))

        # 物品池
        itemPool = list(items_users.index)
        itemNum = len(itemPool)
        self.W = {}

        # IUF 惩罚活跃度高的用户
        for i in range(itemNum - 1):
            i_item = itemPool[i]
            if i_item not in self.W:
                self.W[i_item] = {}
            for j in range(i+1, itemNum):
                j_item = itemPool[j]
                common_user = set(self.itemUser[i_item]) & set(self.itemUser[j_item])
                _num = 0
                for u in common_user:
                    # 计算用户对这两个物品记录的时间衰减数
                    tweak = self.timeweak(self.ratingDict[u][i_item]['time'], self.ratingDict[u][j_item]['time'])
                    _num +=  tweak / math.log(1+len(self.ratingDict[u].keys()))
                i_num = len(self.itemUser[i_item])
                j_num = len(self.itemUser[j_item])
                # 计算两件物品的相似度
                self.W[i_item][j_item] = _num / math.sqrt(i_num * j_num)
                if j_item not in self.W:
                    self.W[j_item] = {}
                self.W[j_item][i_item] = self.W[i_item][j_item]

        # 若为True 则归一化相似度矩阵
        if self.normalized:
            for i in itemPool:
                max_i = sorted(self.W[i].items(), key=lambda x: x[1], reverse=True)[0][1]
                for j in self.W[i].keys():
                    self.W[i][j] = self.W[i][j] / max_i



    def predict(self, item_Idx, item_Idy):
        """
            @function: 预测物品x与物品y的相似度

            @param:
                    item_idx 物品x
                    tem_idy 物品y

            @return: 相似度
        """
        try:
            similarity = self.W[item_Idx][item_Idy]
        except Exception as e:
            similarity = 0
        finally:
            return similarity

    def recommendProducts(self, user, onePick=10, TopN=10):
        """
            @function: 给用户user进行TopN推荐
            @param:
                    user    用户id
                    onePick {number} 每个物品选取onePick个最相似的物品纳入该用户推荐候选列表
                    TopN    {number} 从候选列表中选取TopN为推荐列表
            @return： rank 推荐列表
        """
         # 当前时间戳
        currentTime = int(time.time())
        # 用户历史物品
        user_history = self.ratingDict[user].keys()
        # 初始化推荐候选池
        rankPool = dict()
        # W 拉到局部
        W = self.W
        for history_i in user_history:
            # 与历史物品i最相似的onePick个物品push到rankPool，计算推荐系数
            for j,w_ij in sorted(W[history_i].items(), key=lambda x: x[1], reverse=True)[0:onePick]:
                if j not in rankPool:
                    rankPool[j] = 0
                else:
                    continue
                # 隐反馈数据集  这里的self.ratingDict[user][history_i] 都为1
                tweak = self.timeweak(currentTime, self.ratingDict[user][history_i]["time"])
                rankPool[j] += w_ij * self.ratingDict[user][history_i]["rat"] / (1 + self.beta * tweak)
        rank = sorted(rankPool.items(), key=lambda x: x[1], reverse=True)[0:TopN]
        return rank

if __name__ == "__main__":
    data = pd.read_csv('data/ml-100k/ua.base', sep='\\t', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
    clf = ItemCF_introTime()
    clf.fit(data)
    y_ = clf.recommendProducts(1)
    print(y_)
