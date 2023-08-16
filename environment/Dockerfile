# hash:sha256:3872115565c6100f5e62e9e78db635321e5d391d828e11069152f611d37285f1
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
        python==3.10.9 \
        scikit-learn==1.3.0 \
        seaborn==0.12.2 \
        setuptools==65.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.10.1 \
    nglui==2.14.3 \
    standard-transform==1.4.0
