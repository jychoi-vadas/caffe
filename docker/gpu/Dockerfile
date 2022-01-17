FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

RUN git clone https://github.com/jychoi-vadas/caffe.git . && \
    pip install --upgrade "pip < 9.1.0" && \
    pip install Cython==0.28.1 && \
    pip install numpy==1.11.0 && \
    pip install scipy==0.17.0 && \
    pip install scikit-image==0.13.1 && \
    pip install matplotlib==2.2.2 && \
    pip install ipython==5.6.0 && \
    pip install h5py==2.7.1 && \
    pip install leveldb==0.194 && \
    pip install networkx==2.1 && \
    pip install nose==1.3.7 && \
    pip install pandas==0.22.0 && \
    pip install python-dateutil==1.5 && \
    pip install protobuf==3.5.2 && \
    pip install python-gflags==3.1.2 && \
    pip install pyyaml==3.12 && \
    pip install Pillow==5.1.0 && \
    pip install six==1.11.0
RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="50,52,60,61,70,75" -DCUDA_ARCH_PTX="50,52,60,61,70,75" .. && \
    make -j"$(nproc)" all && \
    make install

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
