FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if any are needed (none required for pure python libraries here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code files
COPY app.py .
COPY generator.py .
COPY templates/ ./templates/

# Expose server port
EXPOSE 8000

# Run uvicorn on 0.0.0.0 to listen for external requests in container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
