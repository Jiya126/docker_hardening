FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY openenv.yaml .
COPY inference.py .
COPY server/ server/
COPY tools/ tools/
COPY tasks/ tasks/
COPY graders/ graders/

ENV USE_MOCK=true
ENV SCANNER=mock
ENV SCA_GYM_TASK=patch_easy
ENV PYTHONPATH="/app:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
