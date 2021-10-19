# If using CPU only replace the following line with:
# FROM ubuntu:18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y python3-dev python3-pip ffmpeg

ARG PROJECT=sova-asr
ARG PROJECT_DIR=/$PROJECT
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# If using CPU only replace the following two lines with:
# RUN pip3 install PuzzleLib
RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/
RUN pip3 install PuzzleLib==1.0.3a0 --install-option="--backend=cuda"

RUN rm -rf $PROJECT_DIR/*

RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
