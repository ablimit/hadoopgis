#! /bin/bash

indexCapacity=1000
fillFactor=0.99

anchor=center
tempPath=/dev/shm/osm/hc

mkdir -p ${tempPath}

echo -e "\n------------------------------------"
echo "building rtree index on test ...."
../genRtreeIndex /data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat ${tempPath}/spatial 20 1000 $fillFactor
rc=$?
if [ $rc -eq 0 ];then
  echo ""
else
  echo -e "\nERROR: genRtreeIndex failed."
  exit $rc ;
fi


for anchor in min center max
do 
  rawdatapath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc/osm.mbb.hc.sorted.${anchor}.dat
  opath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hc/${anchor}
  cut -d" " -f1,2,3,4,5 ${rawdatapath}  > ${tempPath}/hilbert.dat
  ipath=${tempPath}/hilbert.dat


  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    if [ ! -e ${opath}/c${k} ] ;
    then
      mkdir -p ${opath}/c${k}
    fi

    echo "partition size ${k} K"


    ../hilbertPartition ${tempPath}/hilbert.dat ${k} > ${opath}/c${k}/regionmbb.txt
    rc=$?

    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: partition generation failed "
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
    python ../mappartition.py ${tempPath}/pidoid.txt < /data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat > ${opath}/c${k}/osm.part

    rm -f ${tempPath}/pidoid.txt
  
  done
  
  rm -f ${tempPath}/hilbert.dat

done

rm -f ${tempPath}/spatial*

touch done.osm.log

