#! /bin/bash

# jvm params 
jvm="-Xss4m -Xmx100G"

# file path
path=/data2/ablimit/Data/spatialdata/osmout
f1=${path}/planet.1000x1000.dat.1
f2=${path}/europe.1000x1000.dat.2
temp=/dev/shm/temp

# partition parameters
gridsize=11
alpha=0.1
ratio=0.4
sample=0.7
size=100000

if [ ! -e genosmmbb ] ;
then 
    echo "MBR calculator [genosmmbb] is missing."
    exit 0;
fi

if [ ! -e ${temp} ] ;
then
    echo "Temp ${temp} does not exist. Creating it..."
    mkdir -p ${temp}
fi

rm ${temp}/*

echo "calculating MBRs for dataset."
# co-partition the dataset 
# cat ${f1} ${f2} | genosmmbb | python data/normalize.py osm | python data/filterosm.py >  /scratch/aaji/osmmbb.txt
# mv /scratch/aaji/osmmbb.txt /scratch/aaji/osmmbb.txt.bak
# head -n 10000000 /scratch/aaji/osmmbb.txt.bak > /scratch/aaji/osmmbb.txt

echo "calculating the partition size."
lc=`wc -l /scratch/aaji/osmmbb.txt | cut -d' ' -f1 `
p=`expr $((lc/size))`

echo "partition size is: $p"

java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p /scratch/aaji/osmmbb.txt data/partres/osm/osm.${p} ${temp} ${gridsize} ${alpha} ${ratio} ${sample}

