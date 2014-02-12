#! /bin/bash

indexCapacity=1000
fillFactor=0.99

TEMPPATH=`mktemp -d -p /dev/shm`

dir=/data/ablimit/Data/spatialdata/bakup/data
osmdata=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
rdatapath=${dir}/partition/samp/osm
wdatapath=/scratch/ablimit/data/partition/samp/osm

indexPath=/dev/shm

echo -e "\n------------------------------------"
echo "building rtree index on ${osmdata}"
# ../genRtreeIndex ${osmdata} ${tempPath}/spatial 20 1000 $fillFactor
rc=$?
if [ $rc -eq 0 ];then
  echo ""
else
  echo -e "\nERROR: genRtreeIndex failed."
  exit $rc ;
fi


for algo in fg bsp hc slc bos str
do
  if [ ! -e parr/part.${algo} ] ;
  then
    touch parr/part.${algo} ;
    
    for f in 01 05 10 15 20 25 
    do

      for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
      do
        # echo -e "\n------------------------------------"

        # skip if no mbb 
        dest=${rdatapath}/${algo}/f${f}/c${k}
        if [ ! -e ${dest}/regionmbb.txt ] ;
        then
          continue ;
        fi
        # skip if already processed
        #if [ -e ${dest}/osm.part ] ;
        #then
        #  continue ;
        #fi

        wdest=${wdatapath}/${algo}/f${f}/c${k}
        if [ ! -e ${wdest} ] ;
        then
          mkdir -p ${wdest}
        fi

        # start working 
        echo "0.${f} --------- ${k} -------------- ${algo}"

        echo -e "---------------------------------------------"
        echo "generate pid oid mapping ...."
        ../rquery ${dest}/regionmbb.txt ${indexPath}/spatial  > ${TEMPPATH}/pidoid.txt
        rc=$?
        if [ $rc -eq 0 ];then
          echo ""
        else
          echo -e "\nERROR: rqueryfailed."
          exit $rc ;
        fi

        echo -e "\n---------------------------------------------"
        echo "remapping objects"
        python ../mappartition.py ${TEMPPATH}/pidoid.txt < ${osmdata} > ${wdest}/osm.part

        rm -f ${TEMPPATH}/pidoid.txt
      done
    done
  fi
done

rm -rf ${TEMPPATH}

echo -e "\n---------------------------------------------\nDone !"

