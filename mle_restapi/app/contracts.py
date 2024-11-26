from typing import List
from pydantic import BaseModel, field_validator


class ClassificationPayload(BaseModel):
    input_text: str

    class ConfigDict:
        title = "Classification Payload"
        description = "Input data for text classification."

    @field_validator('input_text')
    def check_input_text(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Input text must be a non-empty string")
        return v
    
    @field_validator('input_text')
    def check_numeric_value(cls, v):
        flag = any(c.isalpha() for c in v)
        if not flag:
            raise ValueError("Numeric value cannot be classified")
        return v

class ClassificationResponse(BaseModel):
    label: str
    probability: float

    class ConfigDict:
        title = "Classification Response"
        description = "Response data from text classification api"

    @field_validator('label')
    def check_label_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Label must be a string")
        return v

    @field_validator('probability')
    def check_probability_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v
    
class FeatureVectorResponse(BaseModel):
    features: List[float]

    class ConfigDict:
        title = "FeatureVector Response"
        description = "FeatureVectorResponse from text classification api"
