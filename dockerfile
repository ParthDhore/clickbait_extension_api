# Use an official Python runtime as the base image
FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container
WORKDIR /app

# Copy the Flask application files to the container
COPY . /app

# Install dependencies if any
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the Flask app will run
EXPOSE 5000

# Run the Flask application with multithreading
CMD ["python", "app.py"]
