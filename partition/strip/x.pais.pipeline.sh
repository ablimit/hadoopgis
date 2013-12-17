#! /bin/bash

indexCapacity=1000
fillFactor=0.99

ipath=/data2/ablimit/Data/spatialdata/pais/mbb
opath=/scratch/data/partition/pais/st/x
tempPath=/dev/shm/osm/st/x

mkdir -p ${tempPath}

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do

  echo -e "---------------------------------------------"
  echo "group generating partition region..."

  ../stripGroupPartition ${ipath}/${image}.norm.1.dat 0 20 100 200 400 1000 2000 4000 10000 20000 100000

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
    fi

    echo "partition size ${k} K"

    mv c${k}.txt ${opath}/c${k}/${image}.regionmbb.txt

    python ../simulatecerr.py < ${opath}/c${k}/${image}.regionmbb.txt > ${opath}/c${k}/${image}.idxmbb.gnu

    rc=$?

    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: gnuplot generation failed "
      exit $rc ;
    fi

    echo -e "\n------------------------------------"
    echo "building rtree index on test ...."
    ../genRtreeIndex ${ipath}/${image}.norm.1.dat ${tempPath}/spatial 20 1000 $fillFactor

    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: genRtreeIndex failed."
      exit $rc ;
    fi

    echo -e "---------------------------------------------"
    echo "generate pid oid mapping ...."
    ../rquery ${opath}/c${k}/${image}.regionmbb.txt ${tempPath}/spatial  > ${tempPath}/pidoid.txt
    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: rquery failed."
      exit $rc ;
    fi

    echo -e "\n---------------------------------------------"
    echo "remapping objects"
    python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath}/${image}.norm.1.dat > ${opath}/c${k}/${image}.part

    rm ${tempPath}/spatial*
    rm ${tempPath}/pidoid.txt 
  done
done

touch "done.x.pais.log"
