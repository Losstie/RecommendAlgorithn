from __future__ import division
import pandas as pd
import numpy as np
import math


class UserCF(object):
    def __init__(self, method="base"):
        """
            @function: 算法参数初始化
            @param: method ['base', IIF] 定义计算相似矩阵的方法
        """
        self.method = method
    def fit(self, train):
        """
            @function: 模型训练 得到用户相似矩阵W
            @param:
                train {DataFrame} user item rating 隐反馈数据集有历史行为的item rating都为1
        """
        # 获得列名列表
        self.COLUMNS = train.columns
        # index:userId list:itemIdList
        user_items = train.groupby(self.COLUMNS[0]).agg([list])[self.COLUMNS[1]]
        # 转化为用户-物品-分数字典 {user: {item: rating}}
        self.ratingDict = dict()
        for _, row in train.iterrows():
            user, item, rating = row
            self.ratingDict.setdefault(item,{}).update({user: rating})
        # 创建物品-用户列表字典 {item: [user1，user2]}
        self.userItem = dict(zip(user_items.index, user_items.list))
        # 用户池
        userPool = list(user_items.index)
        userNum = len(userPool)
        self.W = {}

        if self.method == 'base':
            for i in range(userNum - 1):
                if userPool[i] not in self.W:
                    self.W[userPool[i]] = {}
                for j in range(i+1, userNum):
                    common_item = set(self.userItem[userPool[i]]) & set(self.userItem[userPool[j]])
                    i_num = len(self.userItem[userPool[i]])
                    j_num = len(self.userItem[userPool[j]])
                    self.W[userPool[i]][userPool[j]] = len(common_item) / math.sqrt(i_num * j_num)
                    if userPool[j] not in self.W:
                        self.W[userPool[j]] = {}
                    self.W[userPool[j]][userPool[i]] = self.W[userPool[i]][userPool[j]]
        # User-IIF
        elif method == 'IIF':
            for i in range(userNum - 1):
                if userPool[i] not in self.W:
                    self.W[userPool[i]] = {}
                for j in range(i+1, userNum):
                    common_item = set(self.userItem[userPool[i]]) & set(self.userItem[userPool[j]])
                    _num = 0
                    for item in common_item:
                        _num += 1.0 / math.log(1+len(self.ratingDict[item].keys()))
                    i_num = len(self.userItem[userPool[i]])
                    j_num = len(self.userItem[userPool[j]])
                    self.W[userPool[i]][userPool[j]] = _num / math.sqrt(i_num * j_num)
                    if userPool[j] not in self.W:
                        self.W[userPool[j]] = {}
                        self.W[userPool[j]][userPool[i]] = self.W[userPool[i]][userPool[j]]
        else:
            return
    def predict(self, user_A, user_B):
        """
            @function: 预测用户A与用户B的相似度
            @param:
                    user_A 物品A
                    user_B 物品B
            @return: 相似度
        """
        try:
            similarity = self.W[user_A][user_B]
        except Exception as e:
            similarity = 0
        finally:
            return similarity
    def recommendProducts(self, user, onePick=20, TopN=10):
        # 用户历史物品
        user_history = self.userItem[user]
        # 初始化推荐候选池
        rankPool = dict()
        # W 拉到局部
        W = self.W
        # 与用户最相似的onePick个用户，将相似用户中本用户没有记录的商品push到rankPool，计算推荐系数
        for u,w_uv in sorted(W[user].items(), key=lambda x: x[1], reverse=True)[0:onePick]:
            # 相似用户u的评分的物品
            u_items = self.userItem[u]
            for item in u_items:
                if item in user_history:
                    continue
                else:
                    if item not in rankPool:
                        # 隐反馈数据集  这里的self.ratingDict[user][history_i] 都为1
                        rankPool[item] = w_uv * self.ratingDict[item][u]
                    else:
                        rankPool[item] += w_uv * self.ratingDict[item][u]
        rank = sorted(rankPool.items(), key=lambda x: x[1], reverse=True)[0:TopN]
        return rank
