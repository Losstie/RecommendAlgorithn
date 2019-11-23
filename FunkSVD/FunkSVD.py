from __future__ import division
import pandas as pd
import numpy as np
import joblib


class FunkSVD(object):

    def __init__(self, rank=20, penalty=1e-2, lr=1e-3, iterations=5):
        """
            @function: 算法超参初始化
            @param  rank    矩阵分解时对应的低维的维数 [10,200]
            @param penalty  正则化惩罚系数
            @param lr   学习率
            @iterations   迭代最大轮次 [5，20]
        """
        self.rank = rank
        self.penalty = penalty
        self.lr = lr
        self.iterations = iterations

    def fit(self, ratings):
        """
            @function:训练模型
            @param ratings {DataFrame} 评分矩阵 单行形式{user_id, item_id, rating}
        """
        # 得到数据帧 列名列表
        self.COLUMNS = ratings.columns
        self.users_ratings = ratings.groupby(self.COLUMNS[0]).agg([list])[[self.COLUMNS[1], self.COLUMNS[2]]]
        self.items_ratings = ratings.groupby(self.COLUMNS[1]).agg([list])[[self.COLUMNS[0], self.COLUMNS[2]]]
        self.GLOBALMEAN = ratings[self.COLUMNS[2]].mean()
        self.P, self.Q = self.solve(ratings)

    def initMatrix(self):
        """
            @function: 初始化用户和物品隐语义矩阵
            @return:初始化矩阵P，Q
        """
        # User-LatentFactor
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.rank).astype(np.float32)
        ))
        # Item-LatentFactor
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.rank).astype(np.float32)
        ))

        return P, Q

    def solve(self, ratings):
        """
            @function: 随机梯度下降求解P,Q
            @param ratings {dataframe} uid, iid, rating
            @return: 模型参数P,Q
        """
        P, Q = self.initMatrix()
        for iteration in range(self.iterations):
            print("epoch:%d"%iteration)
            residual = []
            for user_id, item_id, rating in ratings.itertuples(index=False):
                p_i = P[user_id]
                q_j = Q[item_id]
                loss = np.float(rating - np.dot(p_i.T, q_j))
                p_i += self.lr * (loss * q_j - self.penalty * p_i)
                q_j += self.lr * (loss * p_i - self.penalty * q_j)

                P[user_id] = p_i
                Q[item_id] = q_j
                residual.append(loss**2)

            print("loss:%.5f"%np.sqrt(np.mean(residual)))
        return P, Q

    def predict(self, user, item):
        """
            @function: 预测user对item的兴趣程度
            @param user {int} user_id
            @param item {int} item_id
            @return preference {float} 用户对物品的感兴趣程度
        """
        if user not in self.P.keys() or item not in self.Q.keys():
            preference = self.GLOBALMEAN
        else:
            preference = np.dot(self.P[user].T, self.Q[item])
        return preference

    def recommendProducts(self, user, TopN):
        """
            @function: 给用户进行TopN推荐
            @param user {int} user_id
            @param TopN {number} 推荐物品数目
            @return rankList {list}
        """
        rank = []
        if user not in self.P.keys():
            # 若用户为新用户，则返回热门推荐。---冷启动，该方法一般
            item_ratings = self.items_ratings
            for item, user_list in item_ratings:
                rank.append((item, len(user_list)))
        else:
            history = self.users_ratings.loc[user][self.COLUMNS[1]]
            for item in self.Q.keys():
                # 过滤掉用户曾购买过的物品
                if item in history:
                    continue
                else:
                    preference = self.predict(user, item)
                    rank.append((item, preference))
        # 倒排物品表返回TopN推荐
        return sorted(rank, key=lambda x:x[1], reverse=True)[0:TopN]


    def recommendUsers(self, item, TopN):
        """
            @function：推荐TopN个对item感兴趣的用户
            @param item {string} item_id
            @param TopN {number} 推荐用户数目
        """
        rank = []
        if item not in self.Q.keys():
            # 若物品为新物品，则推荐购买力强的用户。----冷启动问题。 这里不太合理
            user_ratings = self.users_ratings
            for user, item_list in user_ratings:
                rank.append((user, len(item_list)))
        else:
            for user in self.P.keys():
                # 过滤掉买过该物品的用户
                history = self.users_ratings.loc[user][self.COLUMNS[1]]
                if item in history:
                    continue
                else:
                    preference = self.predict(user, item)
                    rank.append((user, preference))
        return sorted(rank, key=lambda x: x[1], reverse=True)[0:TopN]
