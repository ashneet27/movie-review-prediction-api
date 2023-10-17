import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app.ml.tokenizer import Tokenizer
from app.models.mlModel import MlModel


class MlModelService:
    RELATIVE_PATH_TO_TRAINING_DATA = "../data/labeledTrainData.tsv"
    TRAINING_DATA = pd.read_csv(
        os.path.join(os.path.dirname(__file__), RELATIVE_PATH_TO_TRAINING_DATA),
        delimiter="\t",
    )

    X = TRAINING_DATA["review"].apply(lambda x: " ".join(Tokenizer().tokenize(x)))
    Y = TRAINING_DATA["sentiment"]

    RANDOM_SEED = 42
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )

    VECTORIZER = CountVectorizer(max_features=100000)
    X_TRAIN_VECTORS = VECTORIZER.fit_transform(X_TRAIN)

    MNB = MultinomialNB().fit(X_TRAIN_VECTORS, Y_TRAIN)

    Y_HAT = MNB.predict(VECTORIZER.transform(X_TEST))

    @classmethod
    def modelInfo(cls) -> MlModel:
        return MlModel(accuracy=cls.calculateAccuracy())

    @classmethod
    def calculateAccuracy(cls) -> float:
        return accuracy_score(cls.Y_HAT, cls.Y_TEST)

    @classmethod
    def predict(cls, X: list) -> list:
        return list(cls.MNB.predict(cls.VECTORIZER.transform(X)))