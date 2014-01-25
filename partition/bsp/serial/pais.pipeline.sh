#! /bin/bash

ipath=/shared/data/pais
prog=./bsp

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  echo "stat:${image}"
  for k in 20   100  200  400 1000 2000 4000 10000 20000 100000
    # 0.00001 0.00005 0.0001 0.0002 0.00049 0.00098 0.00197 0.00492 0.00984 0.04922 
  do

    #NUMOFLINES=$(wc -l < "${ipath}/${image}.norm.1.dat")
    #k=$(echo "${NUMOFLINES} * ${f}" | bc -l | cut -d"." -f1)
    #if [ ! "${k}" ] ;then
    #  continue ;
    #fi

    # echo "partition size ${k}"


    # awk '{printf("%d\n",$0+=$0<0?0:0.9)}'
    ${prog} --bucket ${k} --input ${ipath}/${image}.norm.1.dat

    rc=$?
    if [ ! $rc -eq 0 ];then
      echo -e "\nERROR: partition generation failed."
      exit $rc ;
    fi
  done
    echo "------------------------------------"
done

