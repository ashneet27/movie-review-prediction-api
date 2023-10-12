from pydantic import BaseModel


class MlModel(BaseModel):
    accuracy: float