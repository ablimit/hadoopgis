#! /bin/bash

# hilbert curve partition 
algo=hc
aws s3 sync /data2/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo} s3://aaji/data/partitions/pais/${algo} --exclude '*' --include '*pais.geom*'



