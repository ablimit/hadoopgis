#! /bin/bash

ipath=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
opath=/data2/ablimit/Data/spatialdata/bakup/data/prec

TEMPPATH=`mktemp -d -p /dev/shm`
indexPath=/dev/shm
mkdir -p ${tempPath}

for p in 10 12 14 16 18 20 25 30
do
  if [ ! -e parr/occu.${p} ] ;
  then
    touch parr/occu.${p} ;

    for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
    do
      if [ ! -e ${opath}/p${p}/c${k}/regionmbb.txt ] ;
      then
        continue;
      fi

      echo "--------- ${k} --------------"
      echo "generate pid oid mapping ...."
      ../../rquery ${opath}/p${p}/c${k}/regionmbb.txt ${indexPath}/spatial  > ${TEMPPATH}/pidoid.txt
      rc=$?
      if [ $rc -eq 0 ];then
        echo ""
      else
        echo -e "\nERROR: rqueryfailed."
        exit $rc ;
      fi

      echo -e "\n---------------------------------------------"
      echo "remapping objects"
      python ../../mappartition.py ${TEMPPATH}/pidoid.txt < ${ipath} > ${opath}/p${p}/c${k}/osm.part
      rc=$?
      if [ $rc -eq 0 ];then
        echo ""
      else
        echo -e "\nERROR: partition mapping failed."
        exit $rc ;
      fi

      rm -f ${TEMPPATH}/pidoid.txt
    done
  fi
done

rm -rf ${TEMPPATH}

