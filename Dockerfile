FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
