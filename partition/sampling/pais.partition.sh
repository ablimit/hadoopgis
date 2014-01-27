#! /bin/bash

# ipath=/home/aaji/proj/data/osm/osm.mbb.norm.filter.dat
dir=/home/aaji/proj/data/sampling/pais
opath=/home/aaji/proj/data/partition/samp/pais

for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
  for f in 01 05 10 15 20 25 
  do
    ipath=${dir}/${image}.sample.${f}.dat
    
    for k in 20 100  200  400 1000 2000 4000 10000 20000 100000
    do
      for algo in fg hc slc bos str
      do
        echo "0.${f}% --- ${k} ---- ${algo}"

        prog=algo/${algo}


        dest=${opath}/${algo}/f${f}/c${k}

        mkdir -p ${dest}

        b=$(echo "(${k} * 0.${f}+0.5)/1" | bc ) # | cut -d"." -f1)
        if [ $b -eq 0 ] 
        then
          continue ;
        fi

        if [ "${algo}" == "str" ] && [ $b -lt 4 ] ;
        then 
          continue ;
        fi
        echo "${k} x 0.${f} = ${b} "

        ${prog} -i ${ipath} -b ${b} > ${dest}/${image}.regionmbb.txt

        rc=$?
        if [ ! $rc -eq 0 ];then
          echo "ERROR: partition generation failed. "
          exit $rc ;
        fi
      done
    done
  done
done

