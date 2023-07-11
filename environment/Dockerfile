# hash:sha256:159351805f8fad5e91a615edf84865274ff164a1a83d10bb55826ef9e390d544
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        numpy \
    && conda clean -ya
