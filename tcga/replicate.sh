#! /bin/bash

for i in `cat tcga.txt` 
do
  s3cmd cp --progress s3://aaji/data/tcga/${i}.tsv s3://aaji/data/tcga/${i}.2.tsv
done
