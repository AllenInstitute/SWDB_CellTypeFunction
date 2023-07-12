# hash:sha256:dc2ec57d70565273fd8fa808c466fa9cc8a916471d98199adb82462e36b1b25d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:22.11.1-4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.0.2 \
        matplotlib==3.7.2 \
        numpy==1.25.1 \
        seaborn==0.12.2 \
    && conda clean -ya
