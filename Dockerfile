FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir -r /app/requirements-web.txt

COPY . /app

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn topic_search_server:app --host 0.0.0.0 --port ${PORT}"]

