from fastapi.testclient import TestClient
from app.inference import app
import ast
import pytest

INVALID_INPUT_ERROR_RESPONSE_MSG = "Value error, Input text must be a non-empty string"
INVALID_INPUT_SCHEMA_RESPONSE_MSG = "Field required"

# Create a test client using the FastAPI app
client = TestClient(app)


def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.text == 'Application is healthy'


def test_failure_predictions():
    payload = {"input_text": "rocket"}
    response = client.post("/classify", json=payload)
    assert response.status_code == 500



@pytest.mark.parametrize(
    "input, output",
    [
        ({"input_text": ""}, INVALID_INPUT_ERROR_RESPONSE_MSG),
        ({"input":"rocket"}, INVALID_INPUT_SCHEMA_RESPONSE_MSG)
    ]
)
def test_invalid_inputs(input, output):
    response = client.post("/classify", json=input)
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == output


def test_success_predictions():
    with TestClient(app) as client:
        payload = {"input_text": "rocket"}
        response = client.post("/classify", json=payload)
        assert response.status_code == 200
        response_dict = ast.literal_eval(response.text)
        assert "label" in response_dict
        assert "probability" in response_dict
