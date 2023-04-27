from typing import Any, Optional, Union
from pydantic import BaseModel


class TrainApiData(BaseModel):
    model_name: str

class RegisterApiData(BaseModel):
    run_id: str
    model_name: str

class RegisterByNameApiData(BaseModel):
    model_name: str
    model_name: str
    
class TransitionApiData(BaseModel):
    model_name: str
    transition_stage: str
    version: int

class PredictIdApiData(BaseModel):
    input_image: Any
    run_id: str

class PredictNameApiData(BaseModel):
    input_image: Any
    model_name: str

class SaveModelApiData(BaseModel):
    model_name: str

class RegisterSavedModelApiData(BaseModel):
    filename: str
    model_name: str

class RenameModelApiData(BaseModel):
    old_name: str
    new_name: str

class DeleteApiData(BaseModel):
    model_name: str
    model_version: Optional[Union[list[int], int]]  # list | int in python 10
