FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD ./try/Pytorch-UNet/requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .