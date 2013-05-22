#! /bin/bash

if [ ! $# == 1 ]; then
    echo "Usage: $0 [Input File Path] "
    exit
fi


inFile=`basename $1`
echo $inFile

dir=`dirname $1`
echo $dir

indexCapacity=100
leafCapacity=500
fillFactor=0.95
# tempPath=/dev/shm
tempPath=/tmp

echo "generating approxmiation...."
pais/genmbb < ${dir}/${inFile} > ${tempPath}/${inFile}.mbb


echo "building index on data ...."
./genRtreeIndex ${tempPath}/${inFile}.mbb ${tempPath}/${inFile}.idx $indexCapacity $leafCapacity $fillFactor

echo "generating partition ...."
./genPartitionFromIndex ${tempPath}/${inFile}.idx  > ${tempPath}/${inFile}.res

python pais/mappartition.py ${tempPath}/${inFile}.res < ${dir}/${inFile} > ${dir}/${inFile}.part
# mv ${in}.part ${in}

./genPartitionRegionFromIndex  ${tempPath}/${inFile}.idx > ${dir}/${inFile}.regionmbb.txt 2> ${dir}/${inFile}.idxmbb.gnu


rm ${tempPath}/${inFile}*

