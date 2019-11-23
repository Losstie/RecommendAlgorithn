import pandas as pd
import joblib
from BiasSVD import BiasSVD
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for FunkSVD")
parser.add_argument("--rank", type=int, default=20, help="number of latent Factor")
parser.add_argument("--penalty", type=float, default=1e-2, help="L2 regularization parameters")
parser.add_argument("--epochs", type=int, default=15, help="number of training epochs")
a = parser.parse_args()


if __name__ == "__main__":
    if a.mode == "train":
        data = pd.read_csv('../data/movielens/ratings.dat', sep='::', header=None, names=["userId", "movieId", "rating", "timestamp"],engine='python')
        data.drop(columns=["timestamp"], inplace=True)
        clf = FunkSVD(rank=a.rank, penalty=a.penalty, lr=a.lr, iterations=a.epochs)
        clf.fit(data)
        joblib.dump(clf, "BiasSVD_model.m")
    else:
        clf = joblib.load("BiasSVD_model.m")
        clf.predict(1,122)
