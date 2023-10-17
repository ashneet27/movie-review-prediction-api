from app.models.sentiment import Sentiment
from app.services.mlModelService import MlModelService


class ReviewService:

    @classmethod
    def predictReviewSentiment(cls, review: str) -> Sentiment:
        binarySentiment = int(MlModelService.predict([review])[0])
        return Sentiment.POSITIVE if binarySentiment == 1 else Sentiment.NEGATIVE