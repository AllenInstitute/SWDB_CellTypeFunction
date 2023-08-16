# hash:sha256:1b731c39b49e042da2554045020f88aba6111a000d233e7469aec3b1d5225c99
FROM registry.codeocean.allenneuraldynamics.org/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        conda==23.3.1 \
    && conda clean -ya
RUN conda install -y \
        gcc_linux-64 \
        jupyter==1.0.0 \
        jupyterlab==4.0.2 \
        mamba==1.1.0 \
        matplotlib==3.7.2 \
        notebook==6.5.4 \
        numpy==1.25.1 \
        pip==23.0 \
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
