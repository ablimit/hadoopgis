#! /bin/bash

Capacity=1000
fillFactor=0.99


ipath=/data2/ablimit/Data/spatialdata/pais/oligoIII.mbb.dat
# ipath=temp.txt

opath=/mnt/scratch1/aaji/partition/pais/si/rtree

tempPath=/tmp
# tempPath=/dev/shm

# echo "generating approxmiation...."
# pais/genmbb < ${dir}/${inFile} > ${tempPath}/${inFile}.mbb

# for ic in 10 20 50 100 200 500
for ic in 10000 20000 50000 100000
do
    if [ ! -e ${opath}/ic${ic} ] ;
    then 
	mkdir -p ${opath}/ic${ic}
    fi

    echo "Leaf capacity: ${ic}"

    echo -e "\n####################################"
    echo "building index on data ...."
    # ./genRtreeIndexPais ${ipath} ${tempPath}/paisspatial $ic $Capacity $fillFactor
    ./genRtreeIndexPais ${ipath} ${tempPath}/paisspatial 4 $ic $fillFactor
    
    echo -e "\n####################################"
    echo "generating partition region..."
    ./genPartitionRegionFromIndex  ${tempPath}/paisspatial > ${opath}/ic${ic}/regionmbb.txt 2> ${opath}/ic${ic}/idxmbb.gnu

    echo -e "\n####################################"
    echo "generate pid oid mapping ...."
    ./rquery ${opath}/ic${ic}/regionmbb.txt ${tempPath}/paisspatial  > ${tempPath}/paispidoid.txt

    echo -e "\n####################################"
    echo "remapping objects"
    python pais/mappartition.py ${tempPath}/paispidoid.txt < ${ipath} > ${opath}/ic${ic}/pais.part
    
    rm /tmp/paisspatial*
    rm /tmp/paispidoid.txt 
done

