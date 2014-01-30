#! /bin/bash

ipath=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
opath=/data2/ablimit/Data/spatialdata/bakup/data/prec

tempPath=/dev/shm/osm/prec
indexPath=/dev/shm
mkdir -p ${tempPath}

for p in 4 8 10 12 14 16 18 20 25 30
do
  for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
  do
    if [ ! -e ${opath}/p${p}/c${k}/regionmbb.txt ] ;
    then
      continue;
    fi

    echo "--------- ${k} --------------"
    echo "generate pid oid mapping ...."
    ../../rquery ${opath}/p${p}/c${k}/regionmbb.txt ${indexPath}/spatial  > ${tempPath}/pidoid.txt
    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: rqueryfailed."
      exit $rc ;
    fi

    echo -e "\n---------------------------------------------"
    echo "remapping objects"
    python ../../mappartition.py ${tempPath}/pidoid.txt < ${ipath} > ${opath}/p${p}/c${k}/osm.part

    rm -f ${tempPath}/pidoid.txt

  done
done

