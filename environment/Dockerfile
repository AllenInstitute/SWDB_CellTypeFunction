# hash:sha256:e89053ed084391f26d3981516535e26522b37a3229bb50557e33a801457ca109
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        conda==23.3.1 \
    && conda clean -ya
RUN conda install -y \
        jupyter \
        jupyterlab \
        matplotlib \
        notebook \
        numpy \
        pip \
        python-spams \
        scikit-learn==1.3.0 \
        setuptools==65.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.10.1 \
    meshparty==1.16.14 \
    nglui==2.14.3 \
    standard-transform==1.4.0
