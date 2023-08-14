# hash:sha256:eed1cadc5514455cd3dcaa8c62101554872c4884ef3e8d4ecd2d20d623efabb7
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyter==1.0.0 \
        jupyterlab==4.0.2 \
        matplotlib==3.7.2 \
        notebook==6.5.4 \
        numpy==1.25.1 \
        scikit-learn==1.3.0 \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient
