import os
import numpy as np
import nltk
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from app.ml.multinomialNaiveBayes import MultinomialNaiveBayes
from app.ml.tokenizer import Tokenizer
from app.models.sentiment import Sentiment


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


def predictReviewSentiment(review: str) -> Sentiment:
    reviewAsNpArray = np.array([review])
    binarySentiment = int(MNB.predict(reviewAsNpArray)[0])
    return Sentiment.POSITIVE if binarySentiment == 1 else Sentiment.NEGATIVE