FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# GCCやビルド系 + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential pkg-config cmake \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# JAXは「コンテナ内同梱CUDA/cuDNN」を使う版（localじゃない）
RUN pip install --no-cache-dir "jax[cuda12]"

WORKDIR /work
CMD ["/bin/bash"]