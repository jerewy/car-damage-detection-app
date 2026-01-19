FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements_deploy.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app
COPY ./models /code/models

RUN mkdir -p /code/app/static/uploads && chmod 777 /code/app/static/uploads

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
