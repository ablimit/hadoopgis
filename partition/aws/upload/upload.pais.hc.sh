#! /bin/bash

# hilbert curve partition 
algo=bsp
# aws s3 --dryrun sync /data/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo} s3://aaji/data/partitions/pais/${algo} --exclude '*' --include '*pais.geom*'
aws s3 sync /data/ablimit/Data/spatialdata/bakup/data/partition/pais/${algo} s3://aaji/data/partitions/pais/${algo} --exclude '*' --include '*pais.geom*'



