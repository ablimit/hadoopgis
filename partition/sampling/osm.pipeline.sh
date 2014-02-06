#! /bin/bash

indexCapacity=1000
fillFactor=0.99

tempPath=/dev/shm/osm/samp
if [ ! -e ${tempPath} ] ;
then
  mkdir -p ${tempPath};
else
  rm -rf ${tempPath}  ;
  mkdir -p ${tempPath} ;
fi

dir=/data2/ablimit/Data/spatialdata/bakup/data
osmdata=${dir}/osm.mbb.norm.filter.dat
datapath=${dir}/partition/samp/osm

echo -e "\n------------------------------------"
echo "building rtree index on ${osmdata}"
../genRtreeIndex ${osmdata} ${tempPath}/spatial 20 1000 $fillFactor
rc=$?
if [ $rc -eq 0 ];then
  echo ""
else
  echo -e "\nERROR: genRtreeIndex failed."
  exit $rc ;
fi


for algo in fg bsp hc slc bos str
do
  for f in 01 05 10 15 20 25 
  do
    for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
    do
      echo -e "\n------------------------------------"

      dest=${datapath}/${algo}/f${f}/c${k}

      if [ ! -e ${dest}/regionmbb.txt ] ;
      then
        continue ;
      fi

      echo "0.${f} --------- ${k} -------------- ${algo}"
      continue ;

      echo -e "---------------------------------------------"
      echo "generate pid oid mapping ...."
      ../rquery ${dest}/regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt
      rc=$?
      if [ $rc -eq 0 ];then
        echo ""
      else
        echo -e "\nERROR: rqueryfailed."
        exit $rc ;
      fi

      echo -e "\n---------------------------------------------"
      echo "remapping objects"
      python ../mappartition.py ${tempPath}/pidoid.txt < ${osmdata} > ${dest}/osm.part

      rm -f ${tempPath}/pidoid.txt
    done
  done
done

rm -f ${tempPath}/spatial*

echo -e "\n---------------------------------------------\nDone !"

