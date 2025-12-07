FROM python:3.10-slim

# System dependencies required by sentence-transformers and your graph pipeline
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker caches pip step
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend source code last
COPY . .

# Ensure Python sees local modules like "graph/"
ENV PYTHONPATH=/app

# Final startup command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
