FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-lc"]

ARG ENV_NAME=prune_llm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# Minimal OS utilities commonly needed for cloning/evaluation workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda so we can follow the upstream conda-based install recipe.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && conda config --set always_yes yes --set changeps1 no \
    && conda update -n base -c defaults conda \
    && conda clean -afy

# Reproduce wanda/INSTALL.md exactly:
# 1) create conda env with Python 3.9
# 2) install PyTorch CUDA 11.3 stack from conda
# 3) install HF + evaluation packages from pip
RUN conda create -y -n "${ENV_NAME}" python=3.9 && conda clean -afy
RUN conda install -y -n "${ENV_NAME}" \
    pytorch==1.10.1 \
    torchvision==0.11.2 \
    torchaudio==0.10.1 \
    cudatoolkit=11.3 \
    -c pytorch -c conda-forge \
    && conda clean -afy
RUN conda run -n "${ENV_NAME}" pip install --no-cache-dir \
    "numpy<2" \
    "pyarrow<13" \
    "setuptools<81" \
    transformers==4.28.0 \
    "datasets<3.0" \
    wandb \
    sentencepiece \
    accelerate==0.18.0

# Put the project on PYTHONPATH and make the conda env the default Python.
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV PYTHONPATH=/workspace

WORKDIR /workspace
COPY . /workspace

# RUN conda run -n "${ENV_NAME}" pip install --no-cache-dir "datasets<3.0"

CMD ["bash"]