from fastapi import APIRouter
from app.models.mlModel import MlModel
from app.services.mlModelService import MlModelService



router = APIRouter(
    prefix="/mlModel"
)

@router.get("/info", response_model=MlModel)
def getModelInfo():
    return MlModelService.modelInfo()