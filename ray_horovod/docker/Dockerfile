FROM nvidia/cuda:10.1-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=2.3.0
ENV PYTORCH_VERSION=1.8.1
ENV TORCHVISION_VERSION=0.9.1
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.7.8-1+cuda10.1
ENV MXNET_VERSION=1.6.0.post0

# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers
ENV DEBIAN_FRONTEND dialog

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py -x ${HTTP_PROXY} && \
    python get-pip.py && \
    rm get-pip.py

RUN apt update
RUN pip install tqdm
RUN pip install tensorboardX
RUN pip install torchinfo
RUN apt-get install graphviz -y
RUN pip install torchviz
RUN pip install matplotlib
RUN pip install mlflow

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing packaging
#RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} \
#                keras \
#                h5py

RUN PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu101/torch-${PYTORCH_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu101/torchvision-${TORCHVISION_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl
#RUN pip install mxnet-cu101==${MXNET_VERSION}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# # Miniconda
# ARG MINICONDA_VER=4.7.12.1
# ARG MINICONDA_MD5=81c773ff87af5cfac79ab862942ab6b3
# ENV PATH=/opt/conda/bin:${PATH}
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/user/lib:

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh -O miniconda.sh  && \
#     echo "${MINICONDA_MD5} *miniconda.sh" | md5sum -c - && \
#     /bin/bash miniconda.sh -f -b -p /opt/conda && \
#     rm miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 \
         pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN apt install nano
RUN apt install iproute2  iputils-ping -y 
RUN pip install opencv-python
RUN apt install libgl1-mesa-dev -y
RUN apt install libglib2.0-0 -y
RUN pip install imageio
RUN pip install joblib
RUN pip install torchviz
RUN apt install libxkbcommon-x11-0 -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt install python3-tk -y
ENV DEBIAN_FRONTEND dialog

# ホスト側にある公開鍵をイメージ側に登録
COPY ./ssh/id_rsa.pub /root/.ssh/authorized_keys

# コンテナからコンテナにアクセスするためにホスト側にある公開鍵をイメージ側に登録
COPY ./ssh/id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

#RUN pip install segmentation-models-pytorch
RUN pip install scikit-image
#RUN pip install medpy

#RUN apt install make zsh -y

RUN pip install 'ray[tune]'
RUN pip install optuna