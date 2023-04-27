FROM ghcr.io/mlflow/mlflow:v2.3.0
#FROM python:3.9.16-bullseye
RUN apt-get -y update; apt-get -y install curl
RUN apt-get -y install git
RUN pip install mlflow-torchserve
#RUN pip install mlflow[extras] dgl==1.0.1 numpy==1.23.5 packaging==23.1 scipy==1.9.1 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu




