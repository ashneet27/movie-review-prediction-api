import numpy as np

from app.models.sentiment import Sentiment
from app.services.mlModelService import MlModelService


class reviewService:

    @classmethod
    def predictReviewSentiment(cls, review: str) -> Sentiment:
        reviewAsNpArray = np.array([review])
        binarySentiment = int(MlModelService.MNB.predict(reviewAsNpArray)[0])
        return Sentiment.POSITIVE if binarySentiment == 1 else Sentiment.NEGATIVE