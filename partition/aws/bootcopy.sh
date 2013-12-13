#! /bin/bash

cd

hadoop fs -get s3://aaji/scratch/s3cfg .s3cfg

mkdir -p /tmp/lib
mkdir -p /tmp/include

s3cmd sync s3://aaji/scratch/deps/libs/ /tmp/lib/
s3cmd sync s3://aaji/scratch/deps/includes/ /tmp/include/

sudo cp /tmp/lib/* /usr/lib/
sudo cp -r /tmp/include/* /usr/include/


s3cmd get s3://aaji/scratch/awsjoin/resque.cpp ./
s3cmd get s3://aaji/scratch/awsjoin/makefile ./

make
sudo make install


