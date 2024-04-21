# Use Python 3.11.4 as the base image
FROM python:3.11.4

# Add current directory contents into the container at /app
ADD . /app

# Set the working directory to /app
WORKDIR /app

# Install the dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Expose port 5000 to the outside world
EXPOSE 5000

# Set environment variable for Flask app
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]