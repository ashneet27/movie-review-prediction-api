import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.ml.multinomialNaiveBayes import MultinomialNaiveBayes
from app.ml.tokenizer import Tokenizer
from app.models.mlModel import MlModel


class MlModelService:
    RELATIVE_PATH_TO_TRAINING_DATA = "../data/labeledTrainData.tsv"
    TRAINING_DATA = pd.read_csv(os.path.join(os.path.dirname(__file__), RELATIVE_PATH_TO_TRAINING_DATA), delimiter="\t")

    X = TRAINING_DATA["review"].values
    Y = TRAINING_DATA["sentiment"].values
    RANDOM_SEED = 42

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)


    MNB = MultinomialNaiveBayes(
        classes=np.unique(Y),
        tokenizer=Tokenizer()
    ).fit(X_TRAIN, Y_TRAIN)

    Y_HAT: list = None

    @classmethod
    def modelInfo(cls) -> MlModel:
        return MlModel(accuracy=cls.calculateAccuracy())
    
    @classmethod
    def calculateAccuracy(cls) -> float:
        if cls.Y_HAT == None:
            cls.Y_HAT = cls.MNB.predict(cls.X_TEST)
        return accuracy_score(cls.Y_HAT, cls.Y_TEST)