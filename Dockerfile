# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set working directory
WORKDIR /app

# Install system dependencies and cleanup in one layer to keep image size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    default-jdk \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g typescript \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 coderunner \
    && mkdir -p /app/tmp \
    && chown -R coderunner:coderunner /app

# Copy application code
COPY . .
RUN chown -R coderunner:coderunner /app

# Switch to non-root user
USER coderunner

# Create a directory for temporary files
ENV TMPDIR=/app/tmp
ENV PATH="/app:${PATH}"

# Command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT