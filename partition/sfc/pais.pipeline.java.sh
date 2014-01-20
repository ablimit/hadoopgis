#! /bin/bash

indexCapacity=1000
fillFactor=0.99

anchor=center
tempPath=/dev/shm/pais/hc

mkdir -p ${tempPath}

for image in oligoIII.2
do
  echo -e "\n------------------------------------"
  echo "building rtree index on test ...."
  ../genRtreeIndex /home/aaji/temp/mbb/${image}.norm.1.dat ${tempPath}/spatial 20 ${indexCapacity} $fillFactor
  rc=$?
  if [ $rc -eq 0 ];then
    echo ""
  else
    echo -e "\nERROR: genRtreeIndex failed."
    exit $rc ;
  fi

  for anchor in min center max
  do 
    rawdatapath=/home/aaji/temp/hc/${image}.1.${anchor}.dat
    opath=/home/aaji/temp/hc/${anchor}
    cut -d" " -f1,2,3,4,5 ${rawdatapath}  > ${tempPath}/hilbert.dat
    ipath=${tempPath}/hilbert.dat


    #for k in 20 100 200 400 1000 2000 4000 10000 20000 100000
    for k in 100000
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
      python ../mappartition.py ${tempPath}/pidoid.txt < ${tempPath}/hilbert.dat > ${opath}/c${k}/${image}.1.part

      rm -f ${tempPath}/pidoid.txt

    done

    rm -f ${tempPath}/hilbert.dat

  done

  rm -f ${tempPath}/spatial*
done

touch done.pais.log

