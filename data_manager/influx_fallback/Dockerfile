FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY producer.py .
COPY consumer.py .
COPY main.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
