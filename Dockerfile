# Optimized Dockerfile for BentoML Admission Prediction Service
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies in a single layer and clean up
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy only necessary files (exclude unnecessary files)
COPY src/ ./src/
COPY bentofile.yaml .
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash bentoml && \
    chown -R bentoml:bentoml /app
USER bentoml

# Expose the port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -X POST -f http://localhost:3000/health -d "" || exit 1

# Start the service using the correct module path
CMD ["python", "-m", "bentoml", "serve", "src.service_new:service", "--host", "0.0.0.0", "--port", "3000"]
