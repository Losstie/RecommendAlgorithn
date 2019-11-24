# BiasSVD 算法

## 算法原理
`FunkSVD`较为成功的改进版算法。`BiasSVD`假设评分系统包括三部分的偏置因素：一些和用户物品无关的评分因素。用户有一些和物品无关的评分因素，称为用户偏置项。而物品也有一些和用户无关的评分因素，称为物品偏置项。这很好理解，对于乐观的用户来说，它的评分行为普遍偏高，而对批判性用户来说，他的评分记录普遍偏低，即使他们对同一物品的评分相同，但是他们对该物品的喜好程度却并不一样。同理，对物品来说，以电影为例，受大众欢迎的电影得到的评分普遍偏高，而一些烂片的评分普遍偏低，这些因素都是独立于用户或产品的因素，而和用户对产品的的喜好无关。

假设评分系统平均分为*μ*,第*i*个用户的用户偏置项为$b_i$,而第*j*个物品的物品偏置项为$b_j$，则加入了偏置项以后的优化目标函数$J(p_i,q_j)$是这样的:
$$
\underbrace{argmin}_{p_i,q_j}\sum_{i,j}{(m_{ij}-p^T_iq_j-u-b_i-b_j)^2+\lambda({\Arrowvert{p_i}\Arrowvert}^2_2+{\Arrowvert{q_i}\Arrowvert}^2_2+{\Arrowvert{b_i}\Arrowvert}^2_2+{\Arrowvert{b_j}\Arrowvert}^2_2)}
$$
这个优化目标也可以采用梯度下降法求解。和·`FunkSVD`不同的是，此时我们多了两个偏执项$b_i$和 $b_j$，$p_i$和$q_j$的迭代公式和`FunkSVD`类似，只是每一步的梯度导数稍有不同而已。$b_i$和 $b_j$一般可以初始设置为0，然后参与迭代。迭代公式为：

$$
p_i = p_i +\alpha((m_{ij}-p^T_iq_j-u-b_i-b_j)q_j-\lambda{p_i})
$$

$$
q_j = q_j+\alpha((m_{ij}-p^T_iq_j-u-b_i-b_j)p_i-\lambda{q_j})
$$

$$
b_i=b_i+\alpha(m_{ij}-p^T_iq_j-u-b_i-b_j-\lambda{b_i})
$$

$$
b_j=b_j+\alpha(m_{ij}-p^T_iq_j-u-b_i-b_j-\lambda{b_j})
$$

通过迭代我们最终可以得到**P**和**Q**，进而用于推荐。`BiasSVD`增加了一些额外因素的考虑，因此在某些场景会比`FunkSVD`表现好。

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
