#!/usr/bin/env bash
# install Boost package
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/NVIDIA/OpenSeq2Seq
cd OpenSeq2Seq
git checkout ctc-decoders
cd ..
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
./setup.sh
cd ..