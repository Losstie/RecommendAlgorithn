import pandas as pd
import numpy as np
from ItemCF import ItemCF
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--method", type=str, default="base", help="method for calculate similarity")
parser.add_argument("--alpha", type=float, default=0.6, help="param of harryPotter method")
parser.add_argument("--normalized", type=bool, default=True, help="similarity matrix if normalized or not")
parser.add_argument("--modelPath", type=str, default="./itemCf.m",help="model path")
a = parser.parse_args()


if __name__ == "__main__":
    if a.mode == "train":
        data = pd.read_csv('../data/ml-100k/ua.base', sep='\\t', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
        data.drop(columns=["timestamp"], inplace=True)
        clf = ItemCF(method=a.method, alpha=a.alpha, normalized=a.normalized)
        clf.fit(data)
        joblib.dump(clf, a.modelPath)

    else:
        clf = joblib.load(a.modelPath)
        print(clf.recommendProducts(1, 15, 15))
