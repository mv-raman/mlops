
FROM --platform=linux/amd64 docker.ci.artifacts.walmart.com/devtools-docker/looper/looper-python-agent:3.8.16 as builder-base

# install python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ARG DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://sysproxy.wal-mart.com:8080/
ENV https_proxy=http://sysproxy.wal-mart.com:8080/

# RUN apt-get update && apt-get install --no-install-recommends -y tar wget build-essential libssl-dev libffi-dev libsqlite3-dev xz-utils zlib1g-dev postgresql libpq-dev
RUN yum -y install openssl-devel bzip2-devel libffi-devel sqlite-devel xz-devel
RUN yum groupinstall -y "Development Tools"
RUN yum -y install postgresql-devel

RUN wget https://ci.artifacts.walmart.com:443/artifactory/python-releases-generic-release-remote/3.8.13/Python-3.8.13.tgz
RUN tar xzf Python-3.8.13.tgz && cd Python-3.8.13 && ./configure --enable-optimizations && make altinstall

## upgrade pip
RUN pip3.8 install -i https://repository.walmart.com/repository/pypi-proxy/simple/ pip==20.2.1

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.4.2 python3.8 -
ENV PATH="/root/.poetry/bin:$PATH"

WORKDIR /build
