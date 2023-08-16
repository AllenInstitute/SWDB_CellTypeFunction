# hash:sha256:96ecd3486cc3834f2d9467197e62d73943a9d93cf9ece1fabd6319a597077b20
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
        jupyter==1.0.0 \
        jupyterlab \
        matplotlib \
        notebook==6.5.4 \
        numpy \
        pip \
        python==3.10.9 \
        scikit-learn==1.3.0 \
        seaborn==0.12.2 \
        setuptools==65.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.10.1 \
    meshparty==1.16.14 \
    nglui==2.14.3 \
    standard-transform==1.4.0
