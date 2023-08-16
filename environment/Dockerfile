# hash:sha256:6c87d130f96b4e6651b94a77e8f19a7cc92719cef2e7153d7c7ddd64af6893cb
FROM registry.codeocean.allenneuraldynamics.org/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        conda==23.3.1 \
    && conda clean -ya
RUN conda install -y \
        jupyter==1.0.0 \
        jupyterlab==4.0.2 \
        mamba==1.1.0 \
        matplotlib==3.7.2 \
        notebook==6.5.4 \
        numpy==1.25.1 \
        pip==23.0 \
        posix_ipc \
        python==3.10.9 \
        scikit-learn==1.3.0 \
        seaborn==0.12.2 \
        setuptools==65.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.10.1 \
    meshparty \
    nglui==2.14.3 \
    standard-transform==1.4.0
