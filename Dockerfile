FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Reproduce the model and all artefacts at build time.
RUN python run_pipeline.py

EXPOSE 8501 8000

# Default: launch the dashboard. Override for the API:
#   docker run -p 8000:8000 <image> uvicorn src.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
