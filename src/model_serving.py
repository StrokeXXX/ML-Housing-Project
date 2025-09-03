from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/housing-model/production")

@app.post("/predict")
async def predict(features: dict):
    prediction = model.predict([list(features.values())])
    return {"prediction": float(prediction[0])}
