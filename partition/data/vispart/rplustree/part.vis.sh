#! /bin/bash

fillFactor=0.80

idir=../testdata
opath=meta
# tempPath=/dev/shm
tempPath=meta

if [ ! -e ./geRtreeIndex ] || [ ! -e genPartitionRegionFromIndex ] || [ ! -e rquery ] ;
then
    make
fi

if [ ! -d ${tempPath} ];
then 
    mkdir -p ${tempPath}
fi


for tid in 6
do
    ifile="${idir}/test${tid}.obj.txt"

    echo -e "\n####################################"
    echo "building index on test ${tid}  ...."
    ./genRtreeIndex ${ifile} ${tempPath}/spatial 4 4 $fillFactor

    echo -e "\ngenerating partition region..."
    ./genPartitionRegionFromIndex  ${tempPath}/spatial > ${opath}/regionmbb.${tid}.txt 2> ${opath}/idxmbb.${tid}.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./rquery ${opath}/regionmbb.${tid}.txt ${tempPath}/spatial  > ${tempPath}/pidoid.${tid}.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python mappartition.py ${tempPath}/pidoid.${tid}.txt < ${ifile} > ${opath}/part.${tid}.txt
    
    rm ${tempPath}/spatial.idx
    rm ${tempPath}/spatial.dat
done

