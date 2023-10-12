from fastapi import FastAPI
import nltk
from app.models.review import ReviewIn, ReviewOut

from app.services import predictReviewService

app = FastAPI()

@app.post("/predictReview", response_model=ReviewOut)
def predictReview(reviewIn: ReviewIn):
    sentiment = predictReviewService.predictReviewSentiment(reviewIn.review)
    return ReviewOut(**reviewIn.model_dump(), sentiment=sentiment)

if __name__ == "__main__":
    nltk.download("stopwords")