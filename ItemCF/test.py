import pandas as pd
import numpy as np
from ItemCF import ItemCF
import joblib


if __name__ == "__main__":
    # data = pd.read_csv('data/movielens/ratings.dat', sep='::', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
    # data.drop(columns=["timestamp"], inplace=True)
    # clf = ItemCF(normalized=True)
    # clf.fit(data)
    # joblib.dump(clf, "itemCF_normalized_model.m")
    clf = joblib.load("itemCF_normalized_model.m")
    print(clf.recommendProducts(1))
