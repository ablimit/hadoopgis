#! /bin/bash

fillFactor=0.99
ipath=/data2/ablimit/Data/spatialdata/pais/mbb/oligoIII.2.norm.1.dat
opath=/scratch/data/partition/pais/rt # group partition results  
tempPath=/dev/shm/pais/rt

mkdir -p ${tempPath}

for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  if [ ! -e ${opath}/c${k} ] ;
  then
    mkdir -p ${opath}/c${k}

    echo "partition size ${k}"
    echo -e "\n------------------------------------"
    echo "building index on data ...."
    ../genRtreeIndex ${ipath} ${tempPath}/spatial 20 $k $fillFactor

    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: RTree index generation failed "
      exit $rc ;
    fi

    echo -e "\n------------------------------------"
    echo "generating partition region..."
    ../genPartitionFromIndex  ${tempPath}/spatial > ${opath}/c${k}/regionmbb.txt 2> ${opath}/c${k}/idxmbb.gnu
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
    python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/c${k}/pais.part

    rm ${tempPath}/spatial*
    rm ${tempPath}/pidoid.txt 
  fi
done

touch "done.pais.log"

