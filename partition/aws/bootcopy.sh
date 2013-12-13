#! /bin/bash

cd

hadoop fs -get s3://aaji/scratch/s3cfg .s3cfg

sudo s3cmd sync --dry-run s3://aaji/scratch/deps/libs/ /usr/local/lib/
sudo s3cmd sync --dry-run s3://aaji/scratch/deps/includes/* /usr/local/include/
sudo s3cmd sync --drt-run s3://aaji/scratch/deps/bins/resque /usr/local/bin/

