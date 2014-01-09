#! /bin/bash

ipath=/data2/ablimit/Data/spatialdata/pais/hc

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  pos=7
  for corner in min center max 
  do
    if [ ${corner} == "min" ];
    then
      pos=6 
    elif [ ${corner} == "center" ]
    then 
      pos=7 
    elif [ ${corner} == "max" ]
    then 
      pos=8 
    else
      echo "param corner is NOT correct."
    fi

    echo "sorting ${image} with corner ${corner}."

    infile=${ipath}/${image}.1.dat
    ofile=${ipath}/${image}.1.${corner}.dat
    sort -T /dev/shm --numeric-sort --parallel=10 --key=${pos} ${infile} -o ${ofile}

  done
done


echo "done."

