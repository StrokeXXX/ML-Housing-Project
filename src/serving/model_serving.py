from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from .monitoring import track_request

app = FastAPI()
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
@track_request()
async def predict(features: dict):
    # Votre logique de prédiction
    pass
