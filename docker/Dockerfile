FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER jiamingNo1 jiaming19.huang@foxmail.com

COPY sources.list .

RUN mv /etc/apt/sources.list /etc/apt/sources.list.save && \
    mv sources.list /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
        apt-utils \
        build-essential \
        cmake \
        git \
        curl \
        python3-pip \
        libsm6 libxext6 libxrender-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
EXPOSE 6006

RUN python3 -m pip install --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install setuptools

RUN git clone https://github.com/jiamingNo1/Temporal-Shift-Module.git .

RUN pip3 install -r requirements.txt
