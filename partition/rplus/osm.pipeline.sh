#! /bin/bash

indexCapacity=1000
fillFactor=0.99


ipath=/scratch/data/osm.mbb.norm.filter.dat
# ipath=data.snapshot.txt
# ipath=temp.txt
# opath=/scratch/data/partition/osm/rplus
opath=/scratch/data/partition/osm/rplus/gp # group partition results  
tempPath=/dev/shm
# tempPath=/tmp


echo -e "---------------------------------------------"
echo "group generating partition region..."

# done 
# ./rplusGroupPartition ${ipath} 10 20 50 100 200 500

rc=$?
if [ $rc -eq 0 ];then
  echo "group partition finished."
else
  echo -e "\nERROR: genPartitionRegionFromIndex failed."
  exit $rc ;
fi


for k in 10 20 50 100 200 500
do
  if [ ! -e ${opath}/c${k}k ] ;
  then
    mkdir -p ${opath}/c${k}k
  fi
  
  echo "partition size ${k} K"
  
  cp c${k}k.txt ${opath}/c${k}k/regionmbb.txt
  
  python simulatecerr.py < ${opath}/c${k}k/regionmbb.txt > ${opath}/c${k}k/idxmbb.gnu
  
  rc=$?
  
  if [ $rc -eq 0 ];then
    echo ""
  else
    echo -e "\nERROR: gnuplot generation failed "
    exit $rc ;
  fi

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

