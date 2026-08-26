# Fraud-Spike Detector -- containerized Streamlit deployment.
# Works on AWS App Runner, AWS ECS/Fargate, Render, Railway, Fly.io, or any
# platform that runs an arbitrary Docker image on a port.
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost/lightgbm wheels + matplotlib font cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# models/ and results/ are gitignored -- app.py builds them on first request
# if missing (see ensure_artifacts() in app.py), so no build-time training
# step is required here. Uncomment the next two lines to instead bake the
# trained model into the image at build time (slower build, faster cold start):
# RUN python src/train.py
# RUN python src/evaluate.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
