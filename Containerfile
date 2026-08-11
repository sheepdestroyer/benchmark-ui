FROM python:3.14-slim

WORKDIR /app

# Install system dependencies for benchmark scripts and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
