#! /bin/bash

fillFactor=0.99
ipath=/data2/ablimit/Data/spatialdata/pais/mbb
geompath=/data2/ablimit/Data/spatialdata/pais/geom
opath=/scratch/data/partition/pais/rt # group partition results  
tempPath=/dev/shm/pais/rt

mkdir -p ${tempPath}

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
  do
    if [ ! -e ${opath}/c${k} ] ;
    then
      mkdir -p ${opath}/c${k}
    fi

    echo "partition size ${k}"
    echo -e "\n------------------------------------"
    echo "building index on data ...."
    ../genRtreeIndex ${ipath}/${image}.norm.2.dat ${tempPath}/spatial 20 $k $fillFactor

    rc=$?
    if [ $rc -eq 0 ];then
      echo ""
    else
      echo -e "\nERROR: RTree index generation failed "
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
    python ../mappartition.py ${tempPath}/pidoid.txt < ${ipath}/${image}.norm.2.dat > ${opath}/c${k}/${image}.2.part
    
    python ../remappaistogeom.py ${opath}/c${k}/${image}.2.part < ${geompath}/${image}.2.dat > ${opath}/c${k}/${image}.geom.2.dat

    rm ${tempPath}/spatial*
    rm ${tempPath}/pidoid.txt 
  done
done

touch "done.pais.log"

