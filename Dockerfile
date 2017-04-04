FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# to squash: pip install docker-squash
# sudo docker-squash -t riseml/base:latest-squashed riseml/base:latest

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/loca/cuda/lib64

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF && \
    echo "deb http://repos.mesosphere.com/ubuntu xenial main" > /etc/apt/sources.list.d/mesosphere.list && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
        mesos=1.0.1-* build-essential curl git libfreetype6-dev \
        libpng12-dev libzmq3-dev pkg-config python3 python3-dev \
        python python-dev rsync software-properties-common \
        unzip wget default-jre libevent-dev libcurl3 libsvn1 \
        libsasl2-modules && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install kafka protobuf

RUN apt-get update && apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libopencv-dev

RUN apt-get update && apt-get install -y sudo libmimetic-dev libatlas-dev libatlas-base-dev git unzip


#COPY . /caffe_rtpose 
#RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose.git /caffe_rtpose 
#RUN cd /caffe_rtpose && /caffe_rtpose/install_caffe_and_cpm.sh
