from fastapi import FastAPI
from app.models.review import ReviewIn, ReviewOut
from app.models.sentiment import Sentiment

from app.services import predictReviewService

app = FastAPI()

@app.post("/predictReview", response_model=ReviewOut)
def predictReview(reviewIn: ReviewIn):
    return ReviewOut(**reviewIn.model_dump(), sentiment=Sentiment.POSITIVE)