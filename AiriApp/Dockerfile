FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install wheel

COPY ./requirements.txt ./requirements.txt
COPY ./Addition /app/Addition

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

RUN pip install /app/Addition/medimm-main
RUN pip install /app/Addition/screener-main
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./ /app/

