# Use an official Python runtime as a parent image
FROM python:3.10.6-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
# and the libgl1-mesa-dev and libglib2.0-0 packages for the shared libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt

# Set environment variable for video path
ENV VIDEO_PATH=https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run server.py when the container launches
CMD ["python", "server.py"]