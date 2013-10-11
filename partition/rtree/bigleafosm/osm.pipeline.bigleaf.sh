#! /bin/bash

fillFactor=0.99


ipath=/data2/ablimit/Data/spatialdata/osmout/osm.mbb.norm.filter.dat
opath=/mnt/scratch1/aaji/partition/osm/rtree/bigleaf

make

# TEMPPATH=/dev/shm
TEMPPATH=`mktemp -d -p /dev/shm`

ic=4

for lf in 10000 20000 50000 100000 200000 500000
do
    if [ ! -d ${opath}/lf${lf} ] ;
    then 
	mkdir -p ${opath}/lf${lf}
    fi

    echo "Leaf capacity: ${lf}"

    echo -e "\n####################################"
    echo "building index on data ...."
    echo "genRtreeIndex ${ipath} ${TEMPPATH}/spatial $ic $lf $fillFactor"
    ./loader ${ipath} ${TEMPPATH}/spatial $ic $lf $fillFactor
    
    echo -e "\n####################################"
    echo "generating partition region..."
    ./parmbb ${TEMPPATH}/spatial > ${opath}/lf${lf}/regionmbb.txt 2> ${opath}/lf${lf}/idxmbb.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./pquery ${opath}/lf${lf}/regionmbb.txt ${TEMPPATH}/spatial  > ${TEMPPATH}/pidoid.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python ./mappartition.py ${TEMPPATH}/pidoid.txt < ${ipath} > ${opath}/lf${lf}/osm.part
    
    rm ${TEMPPATH}/* 
done

