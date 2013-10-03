#! /bin/bash

leafCapacity=1000
fillFactor=0.99


ipath=/data2/ablimit/Data/spatialdata/osmout/osm.mbb.norm.filter.dat
# ipath=temp.txt
opath=/mnt/scratch1/aaji/partition/osm/rtree
# tempPath=/dev/shm
tempPath=/tmp

# echo "generating approxmiation...."
# pais/genmbb < ${dir}/${inFile} > ${tempPath}/${inFile}.mbb

for ic in 10 20 50 100 200 500
do
    if [ ! -e ${opath}/ic${ic} ] ;
    then 
	mkdir -p ${opath}/ic${ic}
    fi

    echo "Index capacity: ${ic}"

    echo -e "\n####################################"
    echo "building index on data ...."
    ./genRtreeIndexOsm ${ipath} ${tempPath}/spatial $ic $leafCapacity $fillFactor
    
    echo -e "\n####################################"
    echo "generating partition region..."
    ./genPartitionRegionFromIndex  ${tempPath}/spatial > ${opath}/ic${ic}/regionmbb.txt 2> ${opath}/ic${ic}/idxmbb.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./rquery ${opath}/ic${ic}/regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python osm/mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/ic${ic}/osm.part
    
    rm /tmp/spatial*
    rm /tmp/pidoid.txt 
done

