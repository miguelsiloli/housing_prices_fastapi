# Use the official Python image from the Docker Hub
FROM python:3.10.6-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn pyarrow fastparquet
# libgomp1 issues for lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:handler", "--host", "0.0.0.0", "--port", "8000"]
# switch for main:handler for lambda

# ECR commands
# docker build -t fastapi-housing .
# docker tag fastapi-housing:latest 330464977818.dkr.ecr.eu-north-1.amazonaws.com/projects:latest
# docker push 330464977818.dkr.ecr.eu-north-1.amazonaws.com/projects:latest     