#! /bin/bash

# jvm params 
jvm="-Xss4m -Xmx80000M"

# file path
path=/data2/ablimit/Data/spatialdata/osmout
f1=${path}/planet.dat.1
f2=${path}/europe.dat.2

# partition parameters
gridsize=11
alpha=0.1
ratio=0.4
sample=0.7
size=5000000

echo "calculating MBRs for dataset."
# co-partition the dataset 
cat ${f1} ${f2} | genosmmbb | python data/normalize.py osm > /dev/shm/osmmbb.tx

echo "calculating the partition size."
lc=`wc -l /dev/shm/osmmbb.txt | cut -d' ' -f1 `
p=`expr $((lc/size))`



java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p $/dev/shm/mbb.txt data/partres/osm/osmmbb.${p} ${gridsize} ${alpha} ${ratio} ${sample}

