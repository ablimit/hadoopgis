#! /bin/bash

indexCapacity=1000
fillFactor=0.99


ipath=/scratch/data/osm.mbb.norm.filter.dat
# ipath=data.snapshot.txt
# ipath=temp.txt
opath=/scratch/data/partition/osm/rplus
tempPath=/dev/shm
# tempPath=/tmp


for k in 50 100
do
    if [ ! -e ${opath}/c${k}k ] ;
    then 
	mkdir -p ${opath}/c${k}k
    fi

    echo "partition size ${k} K"

    echo -e "\n---------------------------------------------"
    echo "building rplus index on data ...."
    echo "genRplusIndex ${ipath} ${tempPath}/spatial 20 ${k}000 $fillFactor"
    ./genRplusIndex ${ipath} ${tempPath}/spatial 20 ${k}000 $fillFactor
    rc=$?
    if [ $rc -eq 0 ];then
	echo ""
    else
	echo -e "\nERROR: genRplusIndex failed."
	exit $rc ;
    fi

    echo -e "---------------------------------------------"
    echo "generating partition region..."
    ./genPartitionRegionFromIndex  ${tempPath}/spatial > ${opath}/c${k}k/regionmbb.txt 2> ${opath}/c${k}k/idxmbb.gnu
    rc=$?
    if [ $rc -eq 0 ];then
	echo ""
    else
	echo -e "\nERROR: genPartitionRegionFromIndex failed."
	exit $rc ;
    fi

    echo -e "------------------------------------"
    echo -e "\nremove R+ index data."
    rm ${tempPath}/spatial.idx
    rm ${tempPath}/spatial.dat

    echo -e "\n------------------------------------"
    echo "building rtree index on test ...."
    ./genRtreeIndex ${ipath} ${tempPath}/spatial 20 1000 $fillFactor
    rc=$?
    if [ $rc -eq 0 ];then
	echo ""
    else
	echo -e "\nERROR: genRtreeIndex failed."
	exit $rc ;
    fi

    echo -e "---------------------------------------------"
    echo "generate pid oid mapping ...."
    ./rquery ${opath}/c${k}k/regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt
    rc=$?
    if [ $rc -eq 0 ];then
	echo ""
    else
	echo -e "\nERROR: rqueryfailed."
	exit $rc ;
    fi

    echo -e "\n---------------------------------------------"
    echo "remapping objects"
    python mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/c${k}k/osm.part

    rm ${tempPath}/spatial*
    rm ${tempPath}/pidoid.txt 
done

