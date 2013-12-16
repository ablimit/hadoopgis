#! /bin/bash

fillFactor=0.99

ipath=/data2/ablimit/Data/spatialdata/osmout/osm.mbb.norm.filter.dat
opath=/scratch/data/partition/osm/fg
tempPath=/dev/shm/osm/fg

mkdir -p ${tempPath}

for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
    if [ ! -e ${opath}/c${k} ] ;
    then
        mkdir -p ${opath}/c${k}
    fi

    echo "partition size ${k}"
    echo -e "\n------------------------------------"
    echo "building index on data ...."
    ../genRtreeIndex ${ipath} ${tempPath}/spatial 20 2000 ${fillFactor}

    rc=$?
    if [ $rc -eq 0 ];then
        echo ""
    else
        echo -e "\nERROR: RTree index generation failed "
        exit $rc ;
    fi

    echo -e "\n------------------------------------"
    echo "generating partition region..."
    ../fixedgridPartition ${ipath} ${k} > ${opath}/c${k}/regionmbb.txt 
    rc=$?
    if [ $rc -eq 0 ];then
        echo ""
    else
        echo -e "\nERROR: partition generation failed."
        exit $rc ;
    fi

    echo -e "---------------------------------------------"
    echo "generate pid oid mapping ...."
    ../rquery ${opath}/c${k}/regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt
    rc=$?
    if [ $rc -eq 0 ];then
        echo ""
    else
        echo -e "\nERROR: rquery failed."
        exit $rc ;
    fi

    echo -e "\n---------------------------------------------"
    echo "remapping objects"
    python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/c${k}/osm.part

    rm ${tempPath}/spatial*
    rm ${tempPath}/pidoid.txt 
done

touch "done.osm.log"
