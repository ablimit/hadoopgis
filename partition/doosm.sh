#! /bin/bash

f=osm.mbb.norm.filter

echo "$f"
lc=`wc -l /data2/ablimit/Data/spatialdata/osmout/${f}.dat | cut -d' ' -f1 `
p=`expr $((lc/5000000))`

gridsize=11
alpha=0.1
ratio=0.4
sample=0.7


java -Xss4m -Xmx80000M -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f} ${gridsize} ${alpha} ${ratio} ${sample}

