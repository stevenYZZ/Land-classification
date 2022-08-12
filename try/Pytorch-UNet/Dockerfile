FROM harbor.sensingx.local/sxai/pytorch-base

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

ADD ./try/Pytorch-UNet/requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
CMD ["/bin/bash" "-c"]