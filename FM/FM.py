#!/usr/bin/python
"""
@author:losstie
@description: FM模型，进行ranking阶段排序，得出预测item
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

BATCH_SIZE = 32
K = 8
learning_rate = 1e-4
epochs = 48


class MovieLensDataset(Dataset):
    """MovieLens dataset"""
    def __init__(self, csv_file, transform=None):
        """
        Args:
        @param csv_file (string):Path to the csv file
        @param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.movielens = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.movielens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        userID = self.movielens.iloc[idx, 0]
        itemID = self.movielens.iloc[idx, 1]
        target = self.movielens.iloc[idx, 2]
        target = np.array(target).reshape(-1)
        target = torch.tensor(target, dtype=torch.float32)
        userID = np.array(userID).reshape(-1)
        itemID = np.array(itemID).reshape(-1)

        X = self.movielens.iloc[idx, 3:].values
        X = torch.tensor(X, dtype=torch.float32)
        sample = {"X": X,
                  "target": target,
                  "userID": userID,
                  "itemID": itemID
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


# 定义FM模型
class FM_model(torch.nn.Module):
    """FM Model"""
    def __init__(self, n, k):
        super(FM_model, self).__init__()
        self.n = n
        self.k = k
        self.linear = torch.nn.Linear(self.n, 1, bias=True)
        self.v = torch.nn.Parameter(torch.rand(self.n, self.k))

    def fm_layer(self, x):
        # w_i * x_i 线性部分
        linear_part = self.linear(x)
        # pairwise interactions part 1
        inter_part1 = torch.mm(x, self.v)
        # pairwise interactions part 2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        inter_part = 0.5 * torch.sum(torch.sub(torch.pow(inter_part1, 2), inter_part2), dim=1).reshape(-1, 1)
        output = linear_part + inter_part
        return output

    def forward(self, x):
        output = self.fm_layer(x)
        return output


# 自定义数据集
trainset = MovieLensDataset("./train_data")
sample = trainset[0]
# 数据加载
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
fm_model = FM_model(trainset[0]["X"].shape[0], K)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(fm_model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(epochs):

    for i, batch_data in enumerate(trainloader, 0):
        batch_X = batch_data["X"]
        target = batch_data["target"]
        batch_userID = batch_data["userID"]
        batch_itemID = batch_data["itemID"]

        optimizer.zero_grad()
        output = fm_model(batch_X)
        rmse_loss = torch.sqrt(criterion(output, target))
        loss = rmse_loss
        print("epoch {}, step: {}, loss: {}".format(epoch, i, loss))
        loss.backward()
        optimizer.step()

print("train Finished!")

# 保存训练好模型
torch.save(fm_model.state_dict(), "./fm.pt")

# 加载训练好模型
test_model = FM_model(trainset[0]["X"].shape[0], K)
test_model.load_state_dict(torch.load("./fm.pt"))

# 预测召回item评分
recall_data = pd.read_csv("./recall_data")
recall_X = recall_data.iloc[:, 2:].values
recall_main = torch.tensor(recall_data.iloc[:, :2].values, dtype=torch.float32)

recall_X = torch.tensor(recall_X, dtype=torch.float32)
rating_pred = test_model(recall_X)

result = torch.cat((recall_main, rating_pred), dim=1)

result = pd.DataFrame(result.detach().numpy(), columns=["userID", "itemID", "rating"])
result.rating = MinMaxScaler().fit_transform(result.rating.values.reshape(-1, 1))

result.rating = result.rating.apply(lambda x: x * 5)
result.sort_values(by=["userID", "rating"], inplace=True, ascending=False)
print(result[result["userID"] == 1].rating)
result.to_csv("./result.csv", index=False)

test = pd.read_csv("../data/ml-100k/ub.test", header=None, delimiter='\t', names=['userID', 'itemID', 'rating', 'timestamp'], usecols=[0, 1, 2])

print(result[result["userID"] == 1])
print(test[test['userID'] == 1])
