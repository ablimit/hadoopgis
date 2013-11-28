#! /bin/bash

indexCapacity=1000
fillFactor=0.99

# 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062

ipath=/scratch/data/osm.mbb.norm.filter.dat
opath=/scratch/data/partition/osm/rp
tempPath=/dev/shm/osm/rp

mkdir -p ${tempPath}

echo -e "---------------------------------------------"
echo "group generating partition region..."

../rplusGroupPartition ${ipath} 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062

rc=$?
if [ $rc -eq 0 ];then
  echo "group partition finished."
else
  echo -e "\nERROR: genPartitionRegionFromIndex failed."
  exit $rc ;
fi


for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
do
  if [ ! -e ${opath}/c${k} ] ;
  then
    mkdir -p ${opath}/c${k}
  fi
  
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
    echo -e "\nERROR: rqueryfailed."
    exit $rc ;
  fi

  echo -e "\n---------------------------------------------"
  echo "remapping objects"
  python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/c${k}/osm.part

  rm ${tempPath}/spatial*
  rm ${tempPath}/pidoid.txt 
done

touch okay.osm.txt

