from fastapi import APIRouter

from app.models.review import ReviewIn, ReviewOut
from app.services.reviewService import ReviewService


router = APIRouter(
    prefix="/review"
)

@router.post("/predict", response_model=ReviewOut)
def predictReviewSentiment(reviewIn: ReviewIn):
    sentiment = ReviewService.predictReviewSentiment(reviewIn.review)
    return ReviewOut(**reviewIn.model_dump(), sentiment=sentiment)