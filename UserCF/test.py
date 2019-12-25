import pandas as pd
import numpy as np
from UserCF import UserCF
import joblib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--method", type=str, default="base", help="method for calculate similarity")
parser.add_argument("--modelPath", type=str, default="./userCf.m",help="model path")
a = parser.parse_args()


if __name__ == "__main__":
    if a.mode == "train":
        data = pd.read_csv('../data/ml-100k/ua.base', sep='\\t', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
        data.drop(columns=["timestamp"], inplace=True)
        clf = UserCF(method=a.method)
        clf.fit(data)
        joblib.dump(clf, a.modelPath)

    else:
        clf = joblib.load(a.modelPath)
        print(clf.recommendProducts(1, 15, 10))
