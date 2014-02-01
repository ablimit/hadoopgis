#! /bin/bash

for algo in rp rt fg
do
  continue 
  aws s3 sync /data2/ablimit/Data/spatialdata/bakup/data/partition/osm/${algo} s3://aaji/data/partitions/osm/${algo} --exclude '*.txt' --exclude '*.gnu'

done

#algo=st
#aws s3 sync /data2/ablimit/Data/spatialdata/bakup/data/partition/osm/${algo}/x s3://aaji/data/partitions/osm/${algo} --exclude '*.txt' --exclude '*.gnu'

# hilbert curve partition 
algo=hc
aws s3 sync /data2/ablimit/Data/spatialdata/bakup/data/partition/osm/${algo}/center s3://aaji/data/partitions/osm/${algo} --exclude '*' --include '*.dat'



