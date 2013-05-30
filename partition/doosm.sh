#! /bin/bash

f=osm.mbb.norm.filter

echo "$f"
lc=`wc -l /data2/ablimit/Data/spatialdata/osmout/${f}.dat | cut -d' ' -f1 `
p=`expr $((lc/5000000))`


java -Xss4m -Xmx80000M -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f} 
$1 $2 $3 $4

#rtree.sh
#minskew.sh
./minskew.sh $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f}.minskew.txt I 10
#rkHist.sh
./rkHist.sh $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f}.rkHist.txt 0.1
#rv.sh
./rv.sh $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f}.rv.txt 0.4 
#stHist.sh
./stHist.sh $p /data2/ablimit/Data/spatialdata/osmout/${f}.dat data/partres/${f}.sthist.txt 0.5

