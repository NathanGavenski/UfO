FROM ubuntu:22.04

WORKDIR /UfO
COPY . /UfO

# Install dependencies
RUN apt update && apt upgrade -y && \
    apt install -y \
    patchelf \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libopengl0 \
    ninja-build \
    build-essential \
    wget \
    git \
    vim \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install uv

# Dependencies
RUN git clone https://github.com/NathanGavenski/IL-Datasets ../IL-Datasets
RUN uv sync

