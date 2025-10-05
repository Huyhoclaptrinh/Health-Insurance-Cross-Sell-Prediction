
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Add the app directory to the python path
ENV PYTHONPATH="/app"

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . /app

# Define the command to run the clustering analysis
CMD ["sh", "-c", "python3 src/eda_clustering.py && python3 src/compare_clusters.py && python3 src/visualize_clusters.py"]
