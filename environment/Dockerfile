# hash:sha256:8fb29d2166fe47a7da20227fa22a61ec2fd7e8154309e44e6697d6ff285e356e
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
        scikit-learn \
        seaborn==0.12.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==5.7.0
