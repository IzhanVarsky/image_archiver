FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git python3-pip cmake

RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

COPY ./requirements.txt ./requirements.txt
WORKDIR .
RUN pip3 install -r ./requirements.txt

ENTRYPOINT ["./entry.sh"]