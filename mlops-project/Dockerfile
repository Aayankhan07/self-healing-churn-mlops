# Use a lightweight official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set the working directory inside the container
WORKDIR /app

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy configuration, source code, and API folders
COPY params.yaml /app/
COPY src/ /app/src/
COPY api/ /app/api/

# Copy data folder if it exists locally (so pre-trained model is included)
# If not present, the container can still start, and model can be trained or downloaded.
COPY data/ /app/data/

# Expose the API port
EXPOSE 8000

# Command to run the FastAPI app via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
