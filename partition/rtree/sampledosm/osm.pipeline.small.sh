#! /bin/bash

leafCapacity=1000
fillFactor=0.99


ipath=/data2/hoang/Data/osm/sample20/planet_mbb_filtered_sample20.dat
opath=/mnt/scratch1/aaji/partition/osm/rtree/sample20

make

# TEMPPATH=/dev/shm
TEMPPATH=`mktemp -d`

for ic in 10 20 50 100 200 500
do
    if [ ! -d ${opath}/ic${ic} ] ;
    then 
	mkdir -p ${opath}/ic${ic}
    fi

    echo "Index capacity: ${ic}"

    echo -e "\n####################################"
    echo "building index on data ...."
    echo "genRtreeIndex ${ipath} ${TEMPPATH}/spatial $ic $leafCapacity $fillFactor"
    ./loader ${ipath} ${TEMPPATH}/spatial $ic $leafCapacity $fillFactor
    
    echo -e "\n####################################"
    echo "generating partition region..."
    ./parmbb ${TEMPPATH}/spatial > ${opath}/ic${ic}/regionmbb.txt 2> ${opath}/ic${ic}/idxmbb.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./pquery ${opath}/ic${ic}/regionmbb.txt ${TEMPPATH}/spatial  > ${TEMPPATH}/pidoid.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python ./mappartition.py ${TEMPPATH}/pidoid.txt < ${ipath} > ${opath}/ic${ic}/osm.part
    
    rm ${TEMPPATH}/* 
done

