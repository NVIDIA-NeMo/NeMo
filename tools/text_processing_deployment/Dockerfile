# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile for C++ (inverse) text normalization backend Sparrowhawk https://github.com/google/sparrowhawk

# set base image (host OS)
FROM conda/miniconda3

# set the working directory in the container
WORKDIR /workspace

# install dependencies
RUN conda install conda-build -y
RUN apt-get update &&     apt-get install -y --reinstall build-essential &&     apt-get upgrade -y &&     apt-get install -y git &&     apt-get install make
RUN git clone https://github.com/google/re2
RUN cd re2 && make && make install
RUN apt-get install build-essential -y && apt-get install wget -y
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
RUN tar xzvf protobuf-2.5.0.tar.gz
RUN cd protobuf-2.5.0 && ./configure && make && make install && ldconfig
RUN conda install -c conda-forge thrax=1.3.4 -y
RUN git clone https://github.com/yzhang123/sparrowhawk.git
RUN cd sparrowhawk &&  git checkout test &&   apt-get install -y autoconf &&     bash autoreconf && ./configure && make && make install && ldconfig
RUN git clone https://github.com/kward/shunit2.git
RUN echo "DONE"