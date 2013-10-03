#! /bin/bash

leafCapacity=1000
fillFactor=0.99


ipath=/data2/ablimit/Data/spatialdata/pais/pais.mbb.dat
# ipath=temp.txt

opath=/mnt/scratch1/aaji/partition/pais/rtree

tempPath=/tmp
# tempPath=/dev/shm

# echo "generating approxmiation...."
# pais/genmbb < ${dir}/${inFile} > ${tempPath}/${inFile}.mbb

# for ic in 10 20 50 100 200 500
for ic in 10000 20000 50000 100000 200000 500000
do
    if [ ! -e ${opath}/ic${ic}leaf ] ;
    then 
	mkdir -p ${opath}/ic${ic}leaf
    fi

    echo "Leaf capacity: ${ic}"

    echo -e "\n####################################"
    echo "building index on data ...."
    ./genRtreeIndexPais ${ipath} ${tempPath}/paisspatial $ic $leafCapacity $fillFactor
    # ./genRtreeIndexPais ${ipath} ${tempPath}/paisspatial 4 $ic $fillFactor
    
    echo -e "\n####################################"
    echo "generating partition region..."
    ./genPartitionRegionFromIndex  ${tempPath}/paisspatial > ${opath}/ic${ic}leaf/regionmbb.txt 2> ${opath}/ic${ic}leaf/idxmbb.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./rquery ${opath}/ic${ic}leaf/regionmbb.txt ${tempPath}/paisspatial  > ${tempPath}/paispidoid.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python pais/mappartition.py ${tempPath}/paispidoid.txt < ${ipath} > ${opath}/ic${ic}leaf/pais.part
    
    rm /tmp/paisspatial*
    rm /tmp/paispidoid.txt 
done

