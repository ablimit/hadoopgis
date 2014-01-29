#! /bin/bash

indexCapacity=1000
fillFactor=0.99

tempPath=/dev/shm/pi

if [ -e ${tempPath} ] ;
then
  rm -rf ${tempPath}  ;
fi

mkdir -p ${tempPath} ;

dir=/data2/ablimit/Data/spatialdata/pais/mbb
datapath==/data2/ablimit/Data/spatialdata/bakup/data/partition/samp/pais

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  pidata=${dir}/${image}.norm.1.dat
  echo -e "\n------------------------------------"
  echo "building rtree index on test ...."
  ../genRtreeIndex ${pidata} ${tempPath}/spatial 20 1000 $fillFactor
  rc=$?
  if [ $rc -eq 0 ];then
    echo ""
  else
    echo -e "\nERROR: genRtreeIndex failed."
    exit $rc ;
  fi


  for f in 01 05 10 15 20 25 
  do
    for k in 864 4322 8644 17288 43220 86441 172882 432206 864412 4322062
    do
      for algo in fg bsp hc slc bos str
      do
        echo -e "\n------------------------------------"
        echo "0.${f} --------- ${k} -------------- ${algo}"

        dest=${datapath}/${algo}/f${f}/c${k}
        rfile=${dest}/${image}.regionmbb.txt

        if [ ! -e ${rfile} ] ;
        then
          continue ;
        fi

        echo -e "---------------------------------------------"
        echo "generate pid oid mapping ...."
        ../rquery ${rfile} ${tempPath}/spatial  > ${tempPath}/pidoid.txt
        rc=$?
        if [ $rc -eq 0 ];then
          echo ""
        else
          echo -e "\nERROR: rqueryfailed."
          exit $rc ;
        fi

        echo -e "\n---------------------------------------------"
        echo "remapping objects"
        python ../mappartition.py ${tempPath}/pidoid.txt < ${pidata} > ${dest}/${image}.part

        rm -f ${tempPath}/pidoid.txt
      done
    done
  done

  rm -f ${tempPath}/spatial*
done

echo -e "\n---------------------------------------------\nDone !"
