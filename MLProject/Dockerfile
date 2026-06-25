FROM python:3.12.7-slim
WORKDIR /app
COPY . /app
RUN pip install mlflow==2.19.0 pandas scikit-learn
CMD ["python", "modelling.py"]