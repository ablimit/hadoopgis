#! /bin/bash

fillFactor=0.80

idir=../testdata
opath=meta
# tempPath=/dev/shm
tempPath=meta

if [ ! -e ./hilbertPartition ] 
then
    echo "partition exe is missing."
    exit 1;
fi

if [ ! -d ${tempPath} ];
then 
    mkdir -p ${tempPath}
fi

jvm="-Xss4m -Xmx256m"

cp /home/aaji/proj/hadoopgis/xxl/xxlcore/target/*.jar ./

for tid in 2 3 4 5 6 7
do
  ifile="${idir}/test${tid}.obj.txt"
  cut -f1,2,3,4,5 ${ifile} > /tmp/data.txt 

  # calculate the hilbert value 
  java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.Hilbert /tmp/data.txt 4 > /tmp/data.hc.dat

  # sort based on the value
  sort -T /dev/shm --numeric-sort --key=7 /tmp/data.hc.dat | cut -d" " -f1,2,3,4,5 > hilbert.dat



done

