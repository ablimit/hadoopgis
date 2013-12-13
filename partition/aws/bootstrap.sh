#! /bin/bash

sudo apt-get -y install cmake

mkdir boots
hadoop fs -get s3://aaji/scratch/deps/geos-3.4.2.tar.bz2 ./
tar xvf geos-3.4.2.tar.bz2
cd geos-3.4.2
mkdir Release
cd Release
cmake ..
make
sudo make install

cd ~/boots

hadoop fs -get s3://aaji/scratch/deps/spatialindex-src-1.8.1.tar.gz ./
tar xvf spatialindex-src-1.8.1.tar.gz
cd spatialindex-src-1.8.1
mkdir Release
cd Release
cmake ..
make
sudo make install

cd

cd boots
hadoop fs -get s3://aaji/scratch/awsjoin/resque.cpp ./
hadoop fs -get s3://aaji/scratch/awsjoin/makefile ./
make
sudo make install

