#!/usr/bin/python
import pandas as pd
"""
@author: losstie
@description: 构建训练集与召回数据集，进行预处理
"""

train_raw_data = pd.read_csv("../data/ml-100k/ub.base", header=None, delimiter='\t',
                             names=['userID', 'itemID', 'rating'], usecols=[0, 1, 2])

test_raw_data = pd.read_csv("../data/ml-100k/ub.test", header=None, delimiter='\t',
                            names=['userID', 'itemID', 'rating', 'timestamp'])
user_info = pd.read_csv("../data/ml-100k/u.user", header=None, delimiter='|',
                        names=['userID','age', 'gender', 'occupation', 'zip_code'], usecols=[0, 1, 2, 3])
item_info = pd.read_csv("../data/ml-100k/u.item", header=None, delimiter='|',
                        names=['itemID', 'movie_title', 'release_date', 'url',
                               'video_release_date', 'unknown', 'action', 'adventure',
                               'animation', 'children', 'comedy', 'crime', 'documentary',
                               'drama', 'fantasy', 'film-Noir', 'horror',
                               'musical', 'mystery', 'romance', 'sci-Fi',
                               'thriller', 'war', 'western'], encoding='ISO-8859-1', usecols=list(set(range(24)) - set([1, 2, 3, 4])))

train_raw_data = pd.merge(train_raw_data, user_info, on='userID', how='left')
train_raw_data = pd.merge(train_raw_data, item_info, on='itemID', how='left')

# 物品可能无法覆盖全部item， one-hot编码可能不一致。
# 使用一个特殊userID将所有item引入
special = pd.DataFrame([00000]*len(item_info), columns=['userID'])
special['age'] = 23
special['gender'] = 'F'
special['occupation'] = 'executive'
special['rating'] = 1
special = pd.concat([special, item_info], axis=1)
special = special.loc[:, ['userID', 'itemID', 'rating', 'age', 'gender', 'occupation', 'unknown', 'action',
                          'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama',
                          'fantasy', 'film-Noir', 'horror', 'musical', 'mystery', 'romance', 'sci-Fi', 'thriller',
                          'war', 'western']]
print(special.isnull().values.any(), "special")

train_raw_data = pd.concat([train_raw_data, special])
# 物品进行one-hot编码
userID = train_raw_data.loc[:, ['userID']]

# 物品的种类
itemID = train_raw_data['itemID']
itemID_onehot = pd.get_dummies(itemID, prefix='item_rating')
train_raw_data["item_id"] = train_raw_data.loc[:, ['itemID']]
train_data = pd.get_dummies(train_raw_data, columns=['item_id'])

# 性别
train_data = pd.get_dummies(train_data, columns=['gender'])

# 职业
train_data = pd.get_dummies(train_data, columns=['occupation'])

# 用户的历史评分矩阵
ratings = train_raw_data.loc[:, 'rating']
m = itemID_onehot != 1
history_ratings_onehot = itemID_onehot.where(m, ratings, axis=0)
history_ratings_onehot = pd.concat([userID, history_ratings_onehot], axis=1) # 连接用户
history_ratings_onehot = history_ratings_onehot.groupby(['userID']).apply(lambda x: x.sum())
history_ratings_onehot.drop('userID', inplace=True, axis=1)
history_ratings_onehot.reset_index(inplace=True)
print(history_ratings_onehot.isnull().values.any(), "history_ratings")


# 构建训练集
train_data = pd.merge(train_data, history_ratings_onehot, on='userID')

train_data = train_data[train_data['userID'] != 00000]
train_data['user_id'] = train_data.loc[:, ['userID']]
train_data = pd.get_dummies(train_data, columns=['user_id'])
print(train_data.isnull().values.any(), "train_data")
train_data.to_csv('train_data', index=False)

# 召回阶段 构建预测ranking数据集
# 按规则召回   1.将各个分类的电影进行流行度排序 每个分类召回10部
#             2. 每个年龄段进行排序， 每个年龄段召回15部 年龄段：[, 16)--0, [16, 21)--1, [21, 30)--2, [30, 40)--3, [40, 50)--4, [50,~)--5
#             3. 按性别分类排序，分别召回10部
#             4. 按职业分类排序， 分别召回10部

user_history = train_raw_data.loc[:, ['userID', 'itemID', 'age', 'gender', 'occupation', 'unknown','action', 'adventure',
                                      'animation','children', 'comedy', 'crime', 'documentary','drama', 'fantasy',
                                      'film-Noir', 'horror','musical', 'mystery', 'romance', 'sci-Fi','thriller', 'war', 'western']]
user_history['count'] = 1

# 按年龄段召回
age_rank = user_history.loc[:, ['age', 'itemID', 'count']]


def set_age_range(x):
    if x<16:
        ans = 0
    elif (x >= 16) and (x < 21):
        ans = 1
    elif (x >= 21) and (x < 30):
        ans = 2
    elif (x >= 30) and (x < 40):
        ans = 3
    elif (x >=40) and (x < 50):
        ans = 4
    else:
        ans = 5
    return ans


age_rank.age = age_rank.age.apply(set_age_range)
age_rank = age_rank.groupby(['age', 'itemID']).apply(lambda x: x['count'].sum())
age_rank = age_rank.reset_index()
age_rank.rename(columns={0:'count'}, inplace=True)
age_rank.sort_values(['age', 'count'], ascending=[1, 0], inplace=True)
age_rank.reset_index(inplace=True, drop=True)
t1 = age_rank[age_rank['age'] == 0][['age', 'itemID']].head(15)
t2 = age_rank[age_rank['age'] == 1][['age', 'itemID']].head(15)
t3 = age_rank[age_rank['age'] == 2][['age', 'itemID']].head(15)
t4 = age_rank[age_rank['age'] == 3][['age', 'itemID']].head(15)
t5 = age_rank[age_rank['age'] == 4][['age', 'itemID']].head(15)
t6 = age_rank[age_rank['age'] == 5][['age', 'itemID']].head(15)
age_rank = pd.concat([t1, t2, t3, t4, t5, t6], ignore_index=True)

# 按男女召回
gender_rank = user_history.loc[:, ['gender', 'itemID', 'count']]
gender_rank = gender_rank.groupby(['gender', 'itemID']).apply(lambda x:x['count'].sum())
gender_rank = gender_rank.reset_index()
gender_rank.rename(columns={0: 'count'}, inplace=True)
gender_rank.sort_values(['gender','count'], ascending=[1, 0], inplace=True)
gender_rank.reset_index(inplace=True, drop=True)
t1 = gender_rank[gender_rank['gender'] == 'F'][['gender','itemID']].head(15)
t2 = gender_rank[gender_rank['gender'] == 'M'][['gender','itemID']].head(15)
gender_rank = pd.concat([t1, t2], ignore_index=True)

# 按职业分类召回
occupation_rank = user_history.loc[:, ['occupation', 'itemID', 'count']]
occupation_rank = occupation_rank.groupby(['occupation', 'itemID']).apply(lambda x:x['count'].sum())
occupation_rank = occupation_rank.reset_index()
occupation_rank.rename(columns={0: 'count'}, inplace=True)
occupation_rank.sort_values(['occupation', 'count'], ascending=[1, 0], inplace=True)
occupation_rank.reset_index(inplace=True, drop=True)
t1 = occupation_rank[occupation_rank['occupation'] == 'administrator'][['occupation', 'itemID']].head(10)
t2 = occupation_rank[occupation_rank['occupation'] == 'artist'][['occupation', 'itemID']].head(10)
t3 = occupation_rank[occupation_rank['occupation'] == 'doctor'][['occupation', 'itemID']].head(10)
t4 = occupation_rank[occupation_rank['occupation'] == 'educator'][['occupation', 'itemID']].head(10)
t5 = occupation_rank[occupation_rank['occupation'] == 'engineer'][['occupation', 'itemID']].head(10)
t6 = occupation_rank[occupation_rank['occupation'] == 'entertainment'][['occupation', 'itemID']].head(10)
t7 = occupation_rank[occupation_rank['occupation'] == 'executive'][['occupation', 'itemID']].head(10)
t8 = occupation_rank[occupation_rank['occupation'] == 'healthcare'][['occupation', 'itemID']].head(10)
t9 = occupation_rank[occupation_rank['occupation'] == 'homemaker'][['occupation', 'itemID']].head(10)
t10 = occupation_rank[occupation_rank['occupation'] == 'lawyer'][['occupation', 'itemID']].head(10)
t11 = occupation_rank[occupation_rank['occupation'] == 'librarian'][['occupation', 'itemID']].head(10)
t12 = occupation_rank[occupation_rank['occupation'] == 'marketing'][['occupation', 'itemID']].head(10)
t13 = occupation_rank[occupation_rank['occupation'] == 'none'][['occupation', 'itemID']].head(10)
t14 = occupation_rank[occupation_rank['occupation'] == 'other'][['occupation', 'itemID']].head(10)
t15 = occupation_rank[occupation_rank['occupation'] == 'programmer'][['occupation', 'itemID']].head(10)
t16 = occupation_rank[occupation_rank['occupation'] == 'retired'][['occupation', 'itemID']].head(10)
t17 = occupation_rank[occupation_rank['occupation'] == 'salesman'][['occupation', 'itemID']].head(10)
t18 = occupation_rank[occupation_rank['occupation'] == 'scientist'][['occupation', 'itemID']].head(10)
t19 = occupation_rank[occupation_rank['occupation'] == 'student'][['occupation', 'itemID']].head(10)
t20 = occupation_rank[occupation_rank['occupation'] == 'technician'][['occupation', 'itemID']].head(10)
t21 = occupation_rank[occupation_rank['occupation'] == 'writer'][['occupation', 'itemID']].head(10)
occupation_rank = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21],
                            ignore_index=True)

# 按电影分类召回
class_rank = user_history.loc[:, ['unknown','action', 'adventure', 'animation','children', 'comedy', 'crime', 'documentary','drama',
                                  'fantasy','film-Noir', 'horror','musical', 'mystery', 'romance', 'sci-Fi','thriller', 'war',
                                  'western', 'itemID', 'count']]
class_rank = class_rank.groupby(['unknown','action', 'adventure','animation','children', 'comedy', 'crime', 'documentary','drama',
                                  'fantasy','film-Noir', 'horror','musical', 'mystery', 'romance', 'sci-Fi','thriller', 'war',
                                  'western', 'itemID']).apply(lambda x: x['count'].sum())
class_rank = class_rank.reset_index()
class_rank.rename(columns={0: 'count'}, inplace=True)
class_rank.sort_values(['unknown', 'action', 'adventure', 'animation','children', 'comedy', 'crime', 'documentary', 'drama',
                        'fantasy', 'film-Noir', 'horror', 'musical', 'mystery', 'romance', 'sci-Fi','thriller', 'war',
                        'western', 'count'], ascending=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], inplace=True)
t1 = class_rank[class_rank['unknown'] == 1].head(10)
t2 = class_rank[class_rank['action'] == 1].head(10)
t3 = class_rank[class_rank['adventure'] == 1].head(10)
t4 = class_rank[class_rank['animation'] == 1].head(10)
t5 = class_rank[class_rank['children'] == 1].head(10)
t6 = class_rank[class_rank['comedy'] == 1].head(10)
t7 = class_rank[class_rank['crime'] == 1].head(10)
t8 = class_rank[class_rank['documentary'] == 1].head(10)
t9 = class_rank[class_rank['drama'] == 1].head(10)
t10 = class_rank[class_rank['fantasy'] == 1].head(10)
t11 = class_rank[class_rank['film-Noir'] == 1].head(10)
t12 = class_rank[class_rank['horror'] == 1].head(10)
t13 = class_rank[class_rank['musical'] == 1].head(10)
t14 = class_rank[class_rank['mystery'] == 1].head(10)
t15 = class_rank[class_rank['romance'] == 1].head(10)
t16 = class_rank[class_rank['sci-Fi'] == 1].head(10)
t17 = class_rank[class_rank['thriller'] == 1].head(10)
t18 = class_rank[class_rank['war'] == 1].head(10)
t19 = class_rank[class_rank['western'] == 1].head(10)
class_rank = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19], ignore_index=True)
class_rank.drop(['count'], axis=1, inplace=True)

# 构建预测数据集
test_data = user_info
test_data['age_range'] = test_data.age.apply(set_age_range)

# 按年龄召回item匹配用户
age_rank.rename(columns={'age': 'age_range'}, inplace=True)
recall_data1 = pd.merge(test_data, age_rank, on='age_range', how='left')

# 按职业召回
recall_data2 = pd.merge(test_data, occupation_rank, on='occupation', how='left')

# 按性别召回
recall_data3 = pd.merge(test_data, gender_rank, on='gender', how='left')

# 热门item 与用户笛卡尔积
user_id = test_data.loc[:, ['userID']]
length = len(class_rank)
container = ''
for k in user_id.iterrows():
    tmp = k[1].to_list() * length
    tmp = pd.DataFrame(tmp, columns=['userID'])
    tmp = pd.concat([tmp, class_rank], axis=1)
    if type(container) == str:
        container = tmp
    else:
        container = container.append(tmp, ignore_index=True)

# 召回物品可能无法覆盖全部item， one-hot编码可能不一致。
# 使用一个特殊userID将所有item引入
special = pd.DataFrame([00000]*len(item_info), columns=['userID'])
special['age'] = 23
special['gender'] = 'F'
special['occupation'] = 'executive'
special = pd.concat([special, item_info], axis=1)
special = special.loc[:, ['userID', 'age', 'gender', 'occupation', 'itemID']]

# 合并并去重
t1 = pd.concat([recall_data1, recall_data2, recall_data3])
t1 = t1.drop_duplicates()
t1.drop(columns=['age_range'], inplace=True)
print(len(t1))
t2 = pd.merge(test_data, container, on='userID', how='left')
t2 = t2.loc[:, ['userID', 'age', 'gender', 'occupation', 'itemID']]
print(len(t2))
t2 = pd.concat([t1, t2])
print(len(t2))

# 去除用户历史记录item
t3 = train_raw_data.loc[:, ['userID', 'itemID']]
t3 = t3.groupby(['userID']).agg(list)
t3 = t3.reset_index()
t3.rename(columns={'itemID': 'his_item'}, inplace=True)
t4 = pd.merge(t2, t3, on='userID', how='left')
print(len(t4))
t4['flag'] = t4.apply(lambda x: x['itemID'] in x['his_item'], axis=1)
t5 = t4[t4['flag'] == 0]
recall_data = t5.drop(columns=['his_item', 'flag'])
recall_data = pd.concat([recall_data, special])
print(recall_data.head())


# userID  age gender  occupation  itemID
# 处理召回数据集，进行one-hot编码
recall_data = recall_data.loc[:, ['userID', 'itemID', 'age', 'gender', 'occupation']]
recall_data = pd.merge(recall_data, item_info, on='itemID', how='left')

recall_data['item_id'] = recall_data.loc[:, ['itemID']]
recall_data = pd.get_dummies(recall_data, columns=['item_id'])
recall_data = pd.get_dummies(recall_data, columns=['gender'])
recall_data = pd.get_dummies(recall_data, columns=['occupation'])
recall_data = pd.merge(recall_data, history_ratings_onehot, on='userID')
recall_data = recall_data[recall_data['userID'] != 00000]
recall_data['user_id'] = recall_data.loc[:, ['userID']]
recall_data = pd.get_dummies(recall_data, columns=['user_id'])
t = list(train_data.columns)
t.pop(2)
print(t == list(recall_data.columns))

print(recall_data.isnull().values.any(), "recall_data")
recall_data.to_csv('recall_data', index=False)