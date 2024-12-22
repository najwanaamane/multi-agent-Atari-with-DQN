# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file to the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose port 80 for hosting the app
EXPOSE 80

# Command to run the Streamlit app on port 80
CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
