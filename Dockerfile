# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install flask python-dotenv

# Expose port for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD ["python", "core/app.py"]
