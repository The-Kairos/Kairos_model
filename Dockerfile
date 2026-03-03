# Dockerfile for Kairos Audio Pipeline

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Command to run the pipeline (defaulting to all videos in parallel on CPU)
CMD ["python", "-m", "audio_singlecall.main", "--all", "--parallel", "--workers", "2", "--cpu"]
