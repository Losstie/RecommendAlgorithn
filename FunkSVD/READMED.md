# FunkSVD

## 算法原理

详细请见博文：[矩阵分解用于协调过滤推荐——LFM模型](https://www.cnblogs.com/rainbowly/p/11921582.html)

## 代码明细

### 依赖

- pandas
- numpy
- python3.7

### 数据集
基于MovieLens数据集（显性反馈数据集）

### 运行

clone本仓库到本地，在此目录下运行：
- `python test.py --mode train` 训练模型
- `python test.py --mode test` 测试模型
