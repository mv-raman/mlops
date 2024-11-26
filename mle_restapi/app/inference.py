from fastapi import FastAPI, Response, status
from joblib import load
from contracts import ClassificationPayload, ClassificationResponse, FeatureVectorResponse
import uvicorn
from monitoring import instrumentator

app = FastAPI(title="Text Classifier APP", description="Classification API", version="1.0")
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

@app.on_event('startup')
async def load_model() -> None:
    try:
        app.pipeline = load('./models/model.joblib')
        print('model loaded successfully')
    except Exception as e:
        print(f'Failed to load the model: {str(e)}')
        raise SystemExit(1)

@app.get("/health")
def health_check() -> Response:
    return Response(
            status_code=status.HTTP_200_OK,
            content=f"Application is healthy",
        )

categories = [
    'rec.motorcycles',
    'rec.sport.baseball',
    'sci.electronics',
    'sci.space',
    'soc.religion.christian'
    ]

@app.post('/classify')
async def get_prediction(request:ClassificationPayload) -> ClassificationResponse:
    try:
        prediction = categories[app.pipeline.predict([request.input_text]).item()]
        predict_proba = round(max(app.pipeline.predict_proba([request.input_text])[0]),2)
        return ClassificationResponse(label = prediction,  probability = predict_proba)
    except Exception as e:
        error_message = "Model Scoring Failed: " + str(e)
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_message,
        )
    
@app.post('/features')
async def get_feature_vector(request:ClassificationPayload) -> FeatureVectorResponse:
    try:
        feature_vector = app.pipeline['vect'].transform([request.input_text]).toarray().reshape(-1,).tolist()
        return FeatureVectorResponse(features = feature_vector)
    except Exception as e:
        error_message = "Model Scoring Failed: " + str(e)
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_message,
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)