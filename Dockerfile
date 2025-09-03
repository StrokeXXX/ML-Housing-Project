FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
WORKDIR /app
CMD ["uvicorn", "src.model_serving:app", "--host", "0.0.0.0", "--port", "8000"]
