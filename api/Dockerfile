FROM python:3.9.16-bullseye
RUN mkdir /api
RUN mkdir /ml
COPY requirements.txt ./
RUN apt-get update && \
    apt-get install -y curl \
    wget \
    openjdk-11-jre
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY ./api /api
COPY ./ml /ml
CMD ["uvicorn", "api.prova_main:app", "--host", "0.0.0.0", "--port", "8000"]
