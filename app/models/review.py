from pydantic import BaseModel

from app.models.sentiment import Sentiment


class ReviewIn(BaseModel):
    review: str

class ReviewOut(ReviewIn):
    sentiment: Sentiment