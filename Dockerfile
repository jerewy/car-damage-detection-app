# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Install system dependencies for OpenCV
# (Required for cv2 to work on Linux containers)
# Note: libgl1 replaced libgl1-mesa-glx in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install them
# We rename requirements_deploy.txt to requirements.txt inside the container
COPY ./requirements_deploy.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your application code
COPY ./app /code/app
COPY ./models /code/models

# 6. Create a directory for uploads (and set permissions)
# This is CRITICAL for Hugging Face Spaces to allow file saving
RUN mkdir -p /code/app/static/uploads && chmod 777 /code/app/static/uploads

# 7. Command to run the application
# Hugging Face expects the app to run on port 7860 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
