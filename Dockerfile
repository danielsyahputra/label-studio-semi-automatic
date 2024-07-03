FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True \
    PORT=9090

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --preload --bind :$PORT --workers 1 --threads 2 --timeout 0 _wsgi:app                                                             
