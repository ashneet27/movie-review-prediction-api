from fastapi import FastAPI
from app.routers import mlModel, review


app = FastAPI()

app.include_router(review.router)
app.include_router(mlModel.router)