FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RAG_LANGUAGES=hi,ta
ENV PYTHONUNBUFFERED=1
# Sets HF cache to a writable dir inside the Space container
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 7860

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
