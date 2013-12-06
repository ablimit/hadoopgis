#! /bin/bash

indexCapacity=1000
fillFactor=0.99


ipath=/data2/ablimit/Data/spatialdata/pais/mbb/oligoIII.2.norm.1.dat
opath=/scratch/data/partition/pais/st/y
tempPath=/dev/shm/osm/st/y

mkdir -p ${tempPath}

echo -e "---------------------------------------------"
echo "group generating partition region..."

../stripGroupPartition ${ipath} 1 20 100 200 400 1000 2000 4000 10000 20000 100000

rc=$?
if [ $rc -eq 0 ];then
  echo "group partition finished."
else
  echo -e "\nERROR: strip group partition failed."
  exit $rc ;
fi


for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
do
  if [ ! -e ${opath}/c${k} ] ;
  then
    mkdir -p ${opath}/c${k}

    echo "partition size ${k} K"

    cp c${k}.txt ${opath}/c${k}/regionmbb.txt

    python ../simulatecerr.py < ${opath}/c${k}/regionmbb.txt > ${opath}/c${k}/idxmbb.gnu

    rc=$?

    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: gnuplot generation failed "
      exit $rc ;
    fi

    echo -e "\n------------------------------------"
    echo "building rtree index on test ...."
    ../genRtreeIndex ${ipath} ${tempPath}/spatial 20 1000 $fillFactor
    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: genRtreeIndex failed."
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

touch "done.y.pais.log"
